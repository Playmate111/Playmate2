import os
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import deepspeed

import torch.nn.functional as F
from einops import rearrange

import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors import safe_open

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from src.utils.util import get_file_list

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = ["WanModel"]

def attention( 
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None):
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=version,)
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=version,)
    elif SAGE_ATTN_AVAILABLE:
        q = q.unsqueeze(0).transpose(1, 2).to(dtype)
        k = k.unsqueeze(0).transpose(1, 2).to(dtype)
        v = v.unsqueeze(0).transpose(1, 2).to(dtype)
        x = sageattn(q, k, v, dropout_p=dropout_p, is_causal=causal)
        x = x.transpose(1, 2).contiguous()
    else:
        q = q.unsqueeze(0).transpose(1, 2).to(dtype)
        k = k.unsqueeze(0).transpose(1, 2).to(dtype)
        v = v.unsqueeze(0).transpose(1, 2).to(dtype)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).contiguous()

    return x



def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous()
        return out


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))

        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
            dim=-1
        ).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)

    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float() ,
            self.eps
        ).to(origin_dtype)
        return out


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            window_size=self.window_size)

        x = x.flatten(2)
        x = self.o(x)

        return x


class AudioCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.visual_q = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim)
        self.audio_proj = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=True)

    def forward(self, x, encoder_hidden_states):
        B, N, C = x.shape
        q = self.visual_q(x)
        q = q.view(B, N, self.num_heads, self.head_dim)

        a_enc = self.audio_proj(encoder_hidden_states)
        encoder_k, encoder_v = a_enc.view(B, encoder_hidden_states.shape[1], 2, self.num_heads, self.head_dim).unbind(2)
        x = flash_attention(q, encoder_k, encoder_v).flatten(2)

        x = self.out(x)

        return x


class WanI2VCrossAttentionProcessor(nn.Module):
    def __call__(self, attn, x, context, context_lens) -> torch.Tensor:
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), attn.num_heads, attn.head_dim

        q = attn.norm_q(attn.q(x)).view(b, -1, n, d)
        k = attn.norm_k(attn.k(context)).view(b, -1, n, d)
        v = attn.v(context).view(b, -1, n, d)
        k_img = attn.norm_k_img(attn.k_img(context_img)).view(b, -1, n, d)
        v_img = attn.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)

        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = attn.o(x)
        return x


class WanCrossAttentionProcessor(nn.Module):
    def __init__(self, context_dim, hidden_dim, num_heads, use_audio_module=True):
        super().__init__()

        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_audio_module = use_audio_module

        if self.use_audio_module:
            self.visual_audio_attn = AudioCrossAttention(
                dim=self.hidden_dim, encoder_hidden_states_dim=self.context_dim, num_heads=self.num_heads
            )

    def __call__(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        audio_proj: torch.Tensor,
        latents_num_frames: int,
        audio_scale: torch.Tensor,
        org_x=None,
    ) -> torch.Tensor:
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        audio_proj:   [B, 21, L3, C]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), attn.num_heads, attn.head_dim

        q = attn.norm_q(attn.q(x)).view(b, -1, n, d)

        k = attn.norm_k(attn.k(context)).view(b, -1, n, d)
        v = attn.v(context).view(b, -1, n, d)
        k_img = attn.norm_k_img(attn.k_img(context_img)).view(b, -1, n, d)
        v_img = attn.v_img(context_img).view(b, -1, n, d)

        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        img_x = img_x.flatten(2)

        if not self.use_audio_module:
            x = x + img_x

        else:
            x = x + img_x
            x = attn.o(x)
            x = org_x + x

            norm_x = self.visual_audio_attn.norm1(x)
            norm_x = norm_x.view(b * latents_num_frames, -1, n * d)

            audio_proj = audio_proj.contiguous().view(b * latents_num_frames, audio_proj.shape[2], audio_proj.shape[3])
            audio_x = self.visual_audio_attn(norm_x, audio_proj)
            audio_x = audio_x.view(b, -1, n * d)

            x = x + audio_x * audio_scale.view(b, -1, 1)

        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        processor = WanI2VCrossAttentionProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        x,
        context,
        context_lens,
        audio_proj,
        latents_num_frames,
        audio_scale,
        org_x=None,
    ):
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        """
        if audio_proj is None:
            return self.processor(self, x, context, context_lens)
        else:
            return self.processor(
                self,
                x,
                context,
                context_lens,
                audio_proj,
                latents_num_frames,
                audio_scale,
                org_x=org_x,
            )


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        use_audio_module=True,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_audio_module = use_audio_module

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WanI2VCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio_proj,
        latents_num_frames,
        audio_scale,
    ):
        e = (self.modulation + e).chunk(6, dim=1)

        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], grid_sizes, freqs)
        x = x + y * e[2]

        def cross_attn_ffn(
                x, context, context_lens, e, audio_proj, latents_num_frames, audio_scale
        ):
            x = self.cross_attn(
                self.norm3(x),
                context,
                context_lens,
                audio_proj,
                latents_num_frames,
                audio_scale,
                org_x=x,
            )
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]

            return x

        x = cross_attn_ffn(x, context, context_lens, e, audio_proj, latents_num_frames, audio_scale)

        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        e = (
            self.modulation.to(dtype=e.dtype, device=e.device) + e.unsqueeze(1)
        ).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])

        return x


class MLPProj(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class AudioEncModule(nn.Module):
    def __init__(
        self,
        seq_len=5,
        seq_len2=8,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ):
        super().__init__()

        self.context_tokens = context_tokens
        self.output_dim = output_dim
        self.seq_len = seq_len

        self.linear1 = nn.Linear(seq_len * blocks * channels, intermediate_dim)
        self.linear1_2 = nn.Linear(seq_len2 * blocks * channels, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.linear3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.normalize = nn.LayerNorm(output_dim)

    def forward(self, audio_feats):
        audio_latent = rearrange(audio_feats[:, 1:, ...], "b (t f) w l c -> b t f w l c", f=4)
        inter_latent1 = audio_latent[:, :, :1, :self.seq_len // 2 + 1, ...]
        inter_latent1 = rearrange(inter_latent1, "b t f w l c -> b t (f w) l c")
        inter_latent2 = audio_latent[:, :, -1:, self.seq_len // 2:, ...]
        inter_latent2 = rearrange(inter_latent2, "b t f w l c -> b t (f w) l c")
        inter_latent3 = audio_latent[:, :, 1:-1, self.seq_len // 2:self.seq_len // 2 + 1, ...]
        inter_latent3 = rearrange(inter_latent3, "b t f w l c -> b t (f w) l c")
        audio_latent = torch.cat([inter_latent1, inter_latent3, inter_latent2], dim=2)

        audio_feats, audio_latent4 = audio_feats[:, :1, ...], audio_latent

        video_length = audio_feats.shape[1] + audio_latent4.shape[1]
        B, _, _, S, C = audio_feats.shape

        audio_feats = rearrange(audio_feats, "b f w l c -> (b f) w l c")
        batch_size, window_size, blocks, channels = audio_feats.shape
        audio_feats = audio_feats.view(batch_size, window_size * blocks * channels)

        audio_latent4 = rearrange(audio_latent4, "b f w l c -> (b f) w l c")
        batch_size4, window_size4, blocks4, channels4 = audio_latent4.shape
        audio_latent4 = audio_latent4.view(batch_size4, window_size4 * blocks4 * channels4)

        audio_feats = self.linear1(audio_feats)
        audio_feats = rearrange(torch.relu(audio_feats), "(b f) c -> b f c", b=B)
        audio_latent4 = self.linear1_2(audio_latent4)
        audio_latent4 = rearrange(torch.relu(audio_latent4), "(b f) c -> b f c", b=B)
        audio_feats2 = torch.cat([audio_feats, audio_latent4], dim=1)
        batch_size_c, t, C_a = audio_feats2.shape
        audio_feats2 = audio_feats2.view(audio_feats2.shape[0]*audio_feats2.shape[1], audio_feats2.shape[2])
        audio_feats2 = self.linear2(audio_feats2)
        audio_feats2 = torch.relu(audio_feats2)
        out_feat = self.linear3(audio_feats2).reshape(batch_size_c*t, self.context_tokens, self.output_dim)
        out_feat = self.normalize(out_feat)
        out_feat = rearrange(out_feat, "(b f) l c -> b f l c", f=video_length)

        return out_feat


class WanModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=36,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_audio_module=True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_audio_module = use_audio_module

        self.audio_in_dim = 768
        self.audio_proj_dim = 768

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    self.use_audio_module,
                )
                for _ in range(num_layers)
            ]
        )

        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        self.img_emb = MLPProj(1280, dim)

        if self.use_audio_module:
            self.audio_enc_module = AudioEncModule()

        self.init_weights()

        self.set_audio_processor(use_audio_module=self.use_audio_module)

    def forward(
        self,
        latent,
        timestep,
        context,
        clip_fea,
        first_latent,
        audio_feat,
        audio_scale,
        latents_num_frames,
        use_gradient_checkpointing=True
    ):
        weight_dtype = latent.dtype

        if self.use_audio_module:
            audio_proj = self.audio_enc_module(audio_feat)
        else:
            audio_proj = torch.tensor([0])

        if first_latent is not None:
            latent = torch.cat([latent, first_latent], dim=1)

        latent = self.patch_embedding(latent)

        grid_sizes = torch.stack([torch.tensor(u.shape[1:], dtype=torch.long) for u in latent])

        latent = latent.flatten(2).transpose(1, 2)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(weight_dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

        self.freqs = self.freqs.to(latent.device)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block in self.blocks:
            if use_gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                latents = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    latent,
                    e0,
                    grid_sizes,
                    self.freqs,
                    context,
                    context_lens,
                    audio_proj,
                    latents_num_frames,
                    audio_scale,
                    **ckpt_kwargs,
                )
            else:
                kwargs = dict(
                    e=e0,
                    grid_sizes=grid_sizes,
                    freqs=self.freqs,
                    context=context,
                    context_lens=context_lens,
                    audio_proj=audio_proj,
                    latents_num_frames=latents_num_frames,
                    audio_scale=audio_scale,
                )
                latents = block(latent, **kwargs)

            latent = latents

        latent = self.head(latent, e)

        latent = self.unpatchify(latent, grid_sizes)
        latent = torch.stack(latent)

        return latent

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        nn.init.zeros_(self.head.head.weight)

    @property
    def attn_processors(
        self,
    ):
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""copy from https://github.com/XLabs-AI/x-flux/blob/main/src/flux/model.py
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_audio_processor(self, use_audio_module=True):
        attn_procs = {}
        for name in self.attn_processors.keys():
            attn_procs[name] = WanCrossAttentionProcessor(
                context_dim=self.audio_proj_dim,
                hidden_dim=self.dim,
                num_heads=self.num_heads,
                use_audio_module=use_audio_module,
            )
        self.set_attn_processor(attn_procs)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        additional_kwargs={}
    ):
        print(f'******** Load model ********')
        config_file = os.path.join(model_path, "config.json")

        config = cls.load_config(config_file)
        if additional_kwargs == {}:
            model = cls.from_config(config)
        else:
            model = cls.from_config(config, **additional_kwargs)
        model_state_dict = model.state_dict()

        pretrained_state_dict = torch.load(os.path.join(model_path, 'playmate2.pth'))

        new_state_dict = {}
        for k in tqdm(pretrained_state_dict.keys()):
            v = pretrained_state_dict[k]
            if k in model_state_dict:
                if model_state_dict[k].shape != v.shape:
                    print(f'>>> "{k}" shape mismatch! <<<')
                    continue
                new_state_dict[k] = v
            else:
                print(f'>>> miss key "{k}" <<<')

        m, u = model.load_state_dict(new_state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        return model
