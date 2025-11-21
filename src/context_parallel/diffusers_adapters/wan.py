import functools
import torch
from src.playmate2.models.transformer import WanModel, sinusoidal_embedding_1d
from diffusers.utils import logging, scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND

import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import SparseKVAttnMode, UnifiedAttnMode

logger = logging.get_logger(__name__)


def parallelize_transformer(transformer: WanModel, *, mesh=None):
    if getattr(transformer, "_is_parallelized", False):
        return transformer

    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        latent,
        timestep,
        context,
        clip_fea,
        first_latent,
        audio_feat,
        audio_scale,
        latents_num_frames,
    ):
        weight_dtype = latent.dtype

        audio_proj = self.audio_enc_module(audio_feat)

        if first_latent is not None:
            latent = torch.cat([latent, first_latent], dim=1)

        latent = self.patch_embedding(latent)

        grid_sizes = torch.stack([torch.tensor(u.shape[1:], dtype=torch.long) for u in latent])

        latent = latent.flatten(2).transpose(1, 2)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=weight_dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

        self.freqs = self.freqs.to(latent.device)

        e = DP.get_assigned_chunk(e, dim=0, group=batch_mesh)
        e0 = DP.get_assigned_chunk(e0, dim=0, group=batch_mesh)
        grid_sizes = DP.get_assigned_chunk(grid_sizes, dim=0, group=batch_mesh)
        audio_scale = DP.get_assigned_chunk(audio_scale, dim=0, group=batch_mesh)
        latent = DP.get_assigned_chunk(latent, dim=0, group=batch_mesh)
        context = DP.get_assigned_chunk(context, dim=0, group=batch_mesh)
        audio_proj = DP.get_assigned_chunk(audio_proj, dim=0, group=batch_mesh)

        with SparseKVAttnMode(), UnifiedAttnMode(mesh):
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

            for block in self.blocks:
                latents = block(latent, **kwargs)

                latent = latents

            latent = self.head(latent, e)
            latent = self.unpatchify(latent, grid_sizes)
            latent = torch.stack(latent)

        latent = DP.get_complete_tensor(latent, dim=0, group=batch_mesh)

        return latent

    transformer.forward = new_forward.__get__(transformer)

    transformer._is_parallelized = True

    return transformer


def parallelize_pipe(pipe, **kwargs):
    original_call = pipe.__class__.__call__

    @functools.wraps(original_call)
    def new_call(self, **kwargs):
        return original_call(self, **kwargs)

    new_call._is_parallelized = True

    pipe.__class__.__call__ = new_call
    pipe.__class__._is_parallelized = True

    parallelize_transformer(pipe.transformer, **kwargs)

    return pipe
