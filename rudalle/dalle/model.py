# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from einops import rearrange

from .utils import exists, is_empty, init_method_normal

from .transformer import DalleTransformer

import torch.nn as nn

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
import transformers
from typing import Tuple
from torch.cuda.amp import custom_fwd, custom_bwd
%config Completer.use_jedi = False

class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias
 
    def forward(self, input):
        output = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias)
        if self.adapter:
            output += self.adapter(input)
        return output
 
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
 
 
class DequantizeAndLinear(torch.autograd.Function): 
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)
 
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias
 
 
class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
 
    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output 
 
    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"
 
 
def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)
 
    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)
 
 
def convert_to_int8(model):
    """Convert linear and embedding modules to 8-bit with optional adapters"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr(
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )


class DalleModel(torch.nn.Module):
    def __init__(self,
                 device,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 text_seq_length=128,
                 image_tokens_per_dim=32,
                 image_vocab_size=16384,
                 loss_img_weight=7,
                 cogview_sandwich_layernorm=False,
                 cogview_pb_relax=False,
                 cogview_layernorm_prescale=False,
                 custom_relax=False,
                 is_bool_mask=True,
                 mlp_activation='gelu_jit',
                 hf_version='v3'):
        super(DalleModel, self).__init__()
        self.device = device
        self.image_tokens_per_dim = image_tokens_per_dim
        self.image_seq_length = image_tokens_per_dim ** 2
        self.text_seq_length = text_seq_length
        self.total_seq_length = self.text_seq_length + self.image_seq_length
        self.total_vocab_size = vocab_size + image_vocab_size
        self.vocab_size = vocab_size
        self.loss_img_weight = loss_img_weight

        self.hf_version = hf_version

        init_method = init_method_normal(std=0.02)

        self.text_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.image_embeddings = torch.nn.Embedding(image_vocab_size, hidden_size)

        # Position embedding (serial).
        self.text_pos_embeddings = torch.nn.Embedding(text_seq_length + 1, hidden_size)
        self.image_row_embeddings = torch.nn.Embedding(image_tokens_per_dim, hidden_size)
        self.image_col_embeddings = torch.nn.Embedding(image_tokens_per_dim, hidden_size)
        init_method(self.text_pos_embeddings.weight)
        init_method(self.image_row_embeddings.weight)
        init_method(self.image_col_embeddings.weight)

        self.to_logits = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, self.total_vocab_size),
        )

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Transformer
        self.transformer = DalleTransformer(
            num_layers,
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            text_seq_length=text_seq_length,
            image_tokens_per_dim=image_tokens_per_dim,
            cogview_sandwich_layernorm=cogview_sandwich_layernorm,
            cogview_pb_relax=cogview_pb_relax,
            cogview_layernorm_prescale=cogview_layernorm_prescale,
            custom_relax=custom_relax,
            mlp_activation=mlp_activation,
            is_bool_mask=is_bool_mask,
            hf_version=self.hf_version,
        )
        convert_to_int8(self)

    def get_param(self, item):
        return getattr(self, item)

    def get_image_pos_embeddings(self, image_input_ids, past_length=0):
        input_shape = image_input_ids.size()
        row_ids = torch.arange(past_length, input_shape[-1] + past_length,
                               dtype=torch.long, device=self.device) // self.image_tokens_per_dim
        row_ids = row_ids.unsqueeze(0).view(-1, input_shape[-1])
        col_ids = torch.arange(past_length, input_shape[-1] + past_length,
                               dtype=torch.long, device=self.device) % self.image_tokens_per_dim
        col_ids = col_ids.unsqueeze(0).view(-1, input_shape[-1])
        return self.image_row_embeddings(row_ids) + self.image_col_embeddings(col_ids)

    def forward(
            self,
            input_ids,
            attention_mask,
            return_loss=False,
            has_cache=False,
            use_cache=False,
            gradient_checkpointing=None,
    ):
        text = input_ids[:, :self.text_seq_length]
        text_range = torch.arange(self.text_seq_length)
        text_range += (self.vocab_size - self.text_seq_length)
        text_range = text_range.to(self.device)
        text = torch.where(text == 0, text_range, text)
        if self.hf_version == 'v2':
            # some hardcode :)
            text = F.pad(text, (1, 0), value=2)
        text_pos = self.text_pos_embeddings(torch.arange(text.shape[1], device=self.device))
        text_embeddings = self.text_embeddings(text) + text_pos
        image_input_ids = input_ids[:, self.text_seq_length:]

        if exists(image_input_ids) and not is_empty(image_input_ids):
            img_pos = self.get_image_pos_embeddings(image_input_ids, past_length=0)
            image_embeddings = self.image_embeddings(image_input_ids) + img_pos
            embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
        else:
            embeddings = text_embeddings

        if self.hf_version == 'v2':
            # some hardcode :)
            if embeddings.shape[1] > self.total_seq_length:
                embeddings = embeddings[:, :-1]

        alpha = 0.1
        embeddings = embeddings * alpha + embeddings.detach() * (1-alpha)

        attention_mask = attention_mask[:, :, :embeddings.shape[1], :embeddings.shape[1]]
        transformer_output, present_has_cache = self.transformer(
            embeddings, attention_mask,
            has_cache=has_cache, use_cache=use_cache,
            gradient_checkpointing=gradient_checkpointing)

        logits = self.to_logits(transformer_output)
        if return_loss is False:
            return logits, present_has_cache

        labels = torch.cat((text[:, 1:], image_input_ids), dim=1).contiguous().long()
        logits = rearrange(logits, 'b n c -> b c n')

        text_logits = logits[:, :self.vocab_size, :self.text_seq_length].contiguous().float()
        if self.hf_version == 'v3':
            image_logits = logits[:, self.vocab_size:, self.text_seq_length:-1].contiguous().float()
        elif self.hf_version == 'v2':
            image_logits = logits[:, self.vocab_size:, self.text_seq_length:].contiguous().float()
        else:
            raise ValueError(f'Unknown hf_version: {self.hf_version}')

        loss_text = F.cross_entropy(
            text_logits,
            labels[:, :self.text_seq_length])
        loss_img = F.cross_entropy(
            image_logits,
            labels[:, self.text_seq_length:])

        loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
        return loss, {'text': loss_text.data.detach().float(), 'image': loss_img.data.detach().float()}

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)
