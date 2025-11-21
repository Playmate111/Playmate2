import os

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from wan_configs import WAN_CONFIGS
from src.playmate2.pipelines.base import BasePipeline

make_abs_path = lambda fn: os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)


class VideoPipeline(BasePipeline):
    def __init__(
        self,
        text_encoder,
        clip,
        transformer,
        vae,
        scheduler
    ):
        super().__init__()
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.clip = clip
        self.transformer = transformer
        self.vae = vae

    def encode_prompt(self, prompt, device):
        prompt_emb = self.text_encoder([prompt], device)[0].unsqueeze(0)

        return prompt_emb

    def encode_image(self, image, num_frames, height, width, device, msk=None):
        image = self.preprocess_image(image).to(device)
        clip_feat = self.clip.visual([image])

        if msk is not None:
            msk = torch.from_numpy(np.array(msk) / 255)[..., 0]
            msk = F.interpolate(msk.unsqueeze(0).unsqueeze(0), size=(height // 8, width // 8), mode="bicubic")
            msk = msk.repeat(1, num_frames, 1, 1)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)
            with torch.no_grad():
                first_latent = self.vae.encode(
                    torch.concat(
                        [
                            image.transpose(0, 1),
                            torch.zeros(3, num_frames - 1, height, width).to(image.device),
                        ],
                        dim=1,
                    ).unsqueeze(0)
                ).cpu()
            first_latent = torch.concat([msk, first_latent], dim=1)
        else:
            msk = torch.ones(1, num_frames, height // 8, width // 8, device=device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)
            first_latent = self.vae.encode(
                torch.concat(
                    [
                        image.transpose(0, 1),
                        torch.zeros(3, num_frames - 1, height, width).to(image.device),
                    ],
                    dim=1,
                ).unsqueeze(0)
            )
            first_latent = torch.concat([msk, first_latent], dim=1).to(device)

        return clip_feat, first_latent

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_extra_input(self, latents=None):
        return {"seq_len": latents.shape[2] * latents.shape[3] * latents.shape[4] // 4}

    @torch.no_grad()
    def __call__(
        self,
        prompt=WAN_CONFIGS.prompt,
        negative_prompt=WAN_CONFIGS.neg_prompt,
        input_image=None,
        mask=None,
        audio_feat=None,
        denoising_strength=1.0,
        seed=None,
        height=512,
        width=512,
        num_frames=81,
        cfg_scale=5.0,
        audio_cfg_scale=5.0,
        num_inference_steps=30,
        sigma_shift=5.0,
        device='cuda',
        dtype=torch.bfloat16,
        batch_size=3,
        offload=False,
        vad_mask=None
    ):
        if offload:
            self.transformer = self.transformer.cpu()
            torch.cuda.empty_cache()

        self.clip = self.clip.to(device)
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)

        B = batch_size
        audio_feat_len = len(audio_feat)

        B += audio_feat_len - 1

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        context_cache = torch.load(make_abs_path('../../assets/text_cache.pt'))
        prompt_emb = self.encode_prompt(prompt, device)
        prompt_emb = prompt_emb.to(device, dtype)
        if batch_size == 3:
            neg_prompt_emb = context_cache["context_null"]
            neg_prompt_emb = neg_prompt_emb.to(device, dtype)
            b, l, d = prompt_emb.shape
            _, nl, _ = neg_prompt_emb.shape
            if l < nl:
                prompt_emb = torch.cat([prompt_emb, prompt_emb.new_zeros(b, nl - l, d)], dim=1)
            elif l > nl:
                neg_prompt_emb = torch.cat([neg_prompt_emb, neg_prompt_emb.new_zeros(b, l - nl, d)], dim=1)

        image_emb, first_latent = self.encode_image(
            input_image, num_frames, height, width, device, msk=None
        )

        self.clip = self.clip.cpu()
        self.vae = self.vae.cpu()
        self.text_encoder = self.text_encoder.cpu()
        torch.cuda.empty_cache()
        if offload:
            self.transformer = self.transformer.to(device)

        image_emb = image_emb.to(device, dtype)
        first_latent = first_latent.to(device, dtype)

        audio_feat = audio_feat.to(device, dtype)
        if vad_mask is not None:
            vad_mask = vad_mask.to(device, dtype)

        latents_num_frames = (num_frames - 1) // 4 + 1
        total_latents_num_frames = audio_feat.shape[1] // 4 + 1

        all_latents = self.generate_noise(
            (1, 16, total_latents_num_frames, height // 8, width // 8),
            seed=seed,
            device=device,
            dtype=dtype,
        )

        if mask is not None:
            audio_scale_masks = []
            downsample_masks = []

            for i in range(audio_feat_len):
                msk = torch.from_numpy(np.array(mask[i]) / 255)[..., 0]

                downsample_mask = F.interpolate(
                    msk.unsqueeze(0).unsqueeze(0), size=(height // 8, width // 8), mode="bicubic"
                ).unsqueeze(0).to(device)
                downsample_masks.append(downsample_mask)

                audio_scale_mask = F.interpolate(
                    msk.unsqueeze(0).unsqueeze(0), size=(height // 16, width // 16), mode="bicubic"
                ).unsqueeze(0).repeat(1, 1, latents_num_frames, 1, 1).to(device)
                audio_scale_masks.append(audio_scale_mask)

        else:
            downsample_masks = [
                torch.ones((1, 1, 1, height // 8, width // 8)).to(device) for i in range(audio_feat_len)
            ]

            audio_scale_masks = [
                torch.ones((1, 1, latents_num_frames, height // 16, width // 16)).to(device) for i in
                range(audio_feat_len)
            ]

        audio_scale_masks = torch.cat(audio_scale_masks)

        shift = 0
        shift_offset = 7
        overlap = 3

        for progress_id, timestep in enumerate(tqdm(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=device)

            pred_latents = torch.zeros_like(all_latents)
            counter = torch.zeros((1, 1, total_latents_num_frames, 1, 1)).to(device, dtype)

            for batch, index_start in enumerate(range(0, total_latents_num_frames, latents_num_frames - overlap)):
                index_start -= shift

                idx_list = list(range(index_start, index_start + latents_num_frames))
                latents = []
                for idx in idx_list:
                    idx = idx % all_latents.shape[2]
                    latents.append(all_latents[:, :, idx])
                latents = torch.stack(latents, 2)

                idx = idx_list[0] % all_latents.shape[2]
                start_idx = min(idx * 4 + 3, audio_feat.shape[1] - 1)
                chunk_audio_feat = [audio_feat[:, start_idx:start_idx + 1]]
                if vad_mask is not None:
                    chunk_vad_mask = [vad_mask[:, start_idx:start_idx + 1]]
                for idx in idx_list[1:]:
                    idx = idx % all_latents.shape[2]
                    audio_idx = [min(idx * 4 + i, audio_feat.shape[1] - 1) for i in range(4)]
                    chunk_audio_feat.append(audio_feat[:, audio_idx])
                    if vad_mask is not None:
                        chunk_vad_mask.append(vad_mask[:, audio_idx])
                chunk_audio_feat = torch.cat(chunk_audio_feat, 1)
                if vad_mask is not None:
                    chunk_vad_mask = torch.cat(chunk_vad_mask, 1)
                    chunk_vad_mask = chunk_vad_mask[:, ::4]

                latent_input = torch.cat([latents] * B)
                timestep_input = torch.cat([timestep] * B)
                if batch_size == 3:
                    prompt_emb_input = torch.cat([neg_prompt_emb] + [prompt_emb] * (B - 1))
                else:
                    prompt_emb_input = torch.cat([prompt_emb] * B)
                image_emb_input = torch.cat([image_emb] * B)
                first_latent_input = torch.cat([first_latent] * B) if first_latent is not None else None

                audio_feat_input = torch.cat([torch.zeros_like(chunk_audio_feat[:1])] * B)

                audio_scale_input = torch.ones(
                    (B, 1, audio_scale_masks.shape[2], audio_scale_masks.shape[3], audio_scale_masks.shape[4])
                ).to(latents.device, latents.dtype)
                if vad_mask is not None:
                    audio_scale_input[:2] *= chunk_vad_mask.reshape(1, 1, 21, 1, 1)

                audio_feat_input[batch_size - 1:batch_size - 1 + audio_feat_len] = chunk_audio_feat

                noise_pred = self.transformer(
                    latent_input,
                    timestep_input,
                    prompt_emb_input,
                    image_emb_input,
                    first_latent_input,
                    audio_feat_input,
                    audio_scale_input,
                    latents_num_frames,
                )
                noise_preds = noise_pred.chunk(B)

                if batch_size == 3:
                    ua_noise_pred = noise_preds[1]
                    noise_pred = noise_preds[0] + cfg_scale * (noise_preds[1] - noise_preds[0])
                else:
                    ua_noise_pred = noise_preds[0]
                    noise_pred = noise_preds[0]

                for i in range(batch_size - 1, batch_size - 1 + audio_feat_len):
                    noise_pred += audio_cfg_scale * (noise_preds[i] - ua_noise_pred) * downsample_masks[i - batch_size + 1]

                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

                for iii in range(latents_num_frames):
                    p = (index_start + iii) % pred_latents.shape[2]
                    pred_latents[:, :, p] += latents[:, :, iii]
                    counter[:, :, p] += 1

            shift += shift_offset

            pred_latents = pred_latents / counter
            all_latents = pred_latents

        all_latents = torch.cat([first_latent[:, 4:, :1], all_latents], dim=2)
        self.vae = self.vae.to(device)
        frames = self.vae.decode(all_latents)
        frames = self.tensor2video(frames[0][:, 1:])

        return frames
