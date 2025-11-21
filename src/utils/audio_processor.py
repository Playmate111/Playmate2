import os
import os.path
import pyloudnorm as pyln

import torchaudio

import librosa
import numpy as np
import torch
import openunmix.predict as openunmix_predict
import openunmix.utils as openunmix_utils
import openunmix.data as openunmix_data

from einops import rearrange
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from src.utils.wav2vec2 import Wav2Vec2Model as Wav2Vec2ModelV2
from src.utils.util import resample_audio, get_vad_mask


class Separator:
    def __init__(self, device):
        self.device = device
        self.separator = openunmix_utils.load_separator(
            model_str_or_path="umxl",
            targets=None,
            niter=1,
            residual=False,
            wiener_win_len=300,
            device=self.device,
            pretrained=True,
            filterbank="torch",
        )
        self.separator.freeze()
        self.separator.eval()
        self.separator.to(self.device)

    def separate(self, audio_path, vocals_path):
        audio, rate = openunmix_data.load_audio(audio_path)

        with torch.no_grad():
            estimates = openunmix_predict.separate(
                audio=audio,
                rate=rate,
                aggregate_dict=None,
                separator=self.separator,
                device=self.device
            )

        torchaudio.save(
            vocals_path,
            torch.squeeze(estimates['vocals']).to("cpu"),
            sample_rate=self.separator.sample_rate,
        )


class AudioProcessor:
    def __init__(
        self,
        model_path,
        device="cuda:0",
    ) -> None:
        self.device = device
        self.audio_separator = Separator(self.device)

        self.wav2vec = Wav2Vec2ModelV2.from_pretrained(model_path).to(self.device)
        self.wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    def preprocess(
        self,
        wav_file: str,
        vocal_dir,
        sample_rate: int = 16000,
        fps: int = 25,
        get_vad=False
    ):
        vocal_audio_file = os.path.join(vocal_dir, 'vocals.wav')
        os.makedirs(vocal_dir, exist_ok=True)
        self.audio_separator.separate(wav_file, vocal_audio_file)
        vocal_audio_file = resample_audio(vocal_audio_file, os.path.join(vocal_dir, f"vocals-16k.wav"), sample_rate)

        audio_input, sample_rate = librosa.load(vocal_audio_file, sr=sample_rate)  # 采样率为 16kHz

        audio_input = self.loudness_norm(audio_input, sample_rate)

        audio_duration = len(audio_input) / sample_rate
        video_length = audio_duration * 25  # Assume the video fps is 25

        audio_feature = np.squeeze(self.wav2vec_processor(audio_input, sampling_rate=sample_rate).input_values)
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.wav2vec(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        center_indices = torch.arange(0, len(audio_emb), 1,).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_emb.shape[0] - 1)
        audio_emb_split = audio_emb[center_indices][None, ...].cpu().detach()

        if get_vad:
            vad_mask = get_vad_mask(vocal_audio_file, fps=fps).unsqueeze(0).long()
            total_nframe = min(audio_emb_split.shape[1], vad_mask.shape[1])
            audio_emb_split = audio_emb_split[:, :total_nframe]
            vad_mask = vad_mask[:, :total_nframe]

            return audio_emb_split, vad_mask

        else:
            return audio_emb_split

    def loudness_norm(self, audio_array, sr=16000, lufs=-23):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio
