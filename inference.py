import os

import cv2
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = "1"

import argparse
import random
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

import torch

from diffusers.utils import export_to_video
from diffusers.utils import load_image

from src.inference.offload import OffloadConfig
from src.inference.video_infer import VideoInfer
from src.utils.audio_processor import AudioProcessor
from src.utils.util import add_suffix_name, add_audio_to_video
from wan_configs import WAN_CONFIGS
from src.utils.util import get_file_list
from modelscope.pipelines import pipeline

make_abs_path = lambda fn: os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)


def resize_image(image, max_size=768):
    width, height = image.size[0], image.size[1]
    ratio = width / height
    if width > max_size and ratio >= 1.0:
        width, height = max_size, max_size / ratio
    elif height > max_size and ratio < 1.0:
        width, height = max_size * ratio, max_size
    width, height = int(width // 16 * 16), int(height // 16 * 16)
    image = image.resize((width, height))

    return image


def speech_diarization(audio_path, is_chorus=False):
    if is_chorus:
        split_audio_paths = []
        for i in range(args.id_num):
            split_audio_path = f'/tmp/{os.urandom(16).hex()}.wav'
            os.system(f'ffmpeg -i {audio_path} -ac 1 -ar 16000 -y {split_audio_path}')
            split_audio_paths.append(split_audio_path)
    else:
        resample_audio_path = f'/tmp/{os.urandom(16).hex()}.wav'
        os.system(f'ffmpeg -i {audio_path} -ac 1 -ar 16000 -y {resample_audio_path}')

        durations = sd_pipeline(resample_audio_path, oracle_num=args.id_num)['text']
        id_durations = []
        for i in range(args.id_num):
            id_durations.append([duration[:2] for duration in durations if duration[2] == i])

        split_audio_paths = []
        for idx, id_duration in enumerate(id_durations):
            split_audio_path = f'/tmp/{os.urandom(16).hex()}.wav'
            split_audio_paths.append(split_audio_path)
            command = f'''ffmpeg -i {audio_path} -af "volume=enable='not('''
            for duration in id_duration:
                command += f"between(t,{duration[0]},{duration[1]})+"
            command = command[:-1]
            command += f''')':volume=0"  -y {split_audio_path}'''
            os.system(command)

        os.remove(resample_audio_path)

    return split_audio_paths


def process(image_path, audio_path, prompt_path, mask_path, output_video_path, max_size=768):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    torch.cuda.empty_cache()

    image = load_image(image_path)
    image = resize_image(image, max_size=max_size)
    re_width, re_height = image.size

    if os.path.exists(mask_path):
        masks = []
        for fm_path in get_file_list(mask_path, suffix='.png'):
            mask = cv2.imread(fm_path)
            mask = cv2.resize(mask, (re_width, re_height), interpolation=cv2.INTER_NEAREST)
            masks.append(Image.fromarray(mask))
    else:
        masks = None

    if args.id_num > 1:
        if os.path.isfile(audio_path):
            split_audio_paths = speech_diarization(audio_path, is_chorus=args.is_chorus)
            join_audio_path = audio_path
        else:
            split_audio_paths = [
                split_audio_path for split_audio_path in get_file_list(audio_path) if 'split_' in split_audio_path
            ]
            join_audio_path = os.path.join(audio_path, 'join.WAV')
    else:
        split_audio_paths = [audio_path]
        join_audio_path = audio_path

    audio_embs = []
    vad_masks = []
    min_len = 99999
    for split_audio_path in split_audio_paths:
        video_name = os.path.basename(split_audio_path).replace(".wav", "")
        audio_emb, vad_mask = audio_processor.preprocess(
            split_audio_path, f"./vocals/{video_name}", fps=args.fps, get_vad=True
        )

        if audio_emb.shape[1] < min_len:
            min_len = audio_emb.shape[1]
        audio_embs.append(audio_emb)
        vad_masks.append(vad_mask)
    audio_embs = torch.cat([audio_emb[:, :min_len] for audio_emb in audio_embs])
    vad_masks = torch.cat([vad_mask[:, :min_len] for vad_mask in vad_masks])
    vad_mask = vad_masks[:1]
    for vm in vad_masks[1:]:
        vad_mask = vad_mask | vm

    with open(prompt_path, 'r') as f:
        prompt = f.read()
    f.close()

    kwargs = {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "height": re_height,
        "width": re_width,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "cfg_scale": args.guidance_scale,
        "audio_cfg_scale": args.audio_guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "batch_size": 3,
        "offload": True,
        "input_image": image,
        "mask": masks,
        "audio_feat": audio_embs,
        "vad_mask": vad_mask
    }

    output = predictor.inference(kwargs)
    silence_video = add_suffix_name(output_video_path, '_silence')
    add_audio_video = add_suffix_name(output_video_path, '_wa')
    export_to_video(output, silence_video, fps=args.fps)
    add_audio_to_video(silence_video, join_audio_path, add_audio_video)
    if args.fps != 25:
        os.system(
            f'ffmpeg -loglevel quiet -i {add_audio_video} -ss 0.32 -r 25 -vcodec libx264 -pix_fmt yuv420p -y {output_video_path}'
        )
    else:
        os.system(
            f'ffmpeg -loglevel quiet -i {add_audio_video} -ss 0.32 -vcodec libx264 -pix_fmt yuv420p -y {output_video_path}'
        )
    os.remove(silence_video)
    os.remove(add_audio_video)

    if args.id_num > 1 and os.path.isfile(audio_path):
        for split_audio_path in split_audio_paths:
            os.remove(split_audio_path)

    print(f"Video generated successfully: {output_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='pretrained_weights/playmate2')
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--audio_guidance_scale", type=float, default=4.0)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default=WAN_CONFIGS.prompt)
    parser.add_argument("--negative_prompt", type=str, default=WAN_CONFIGS.neg_prompt)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max_size", type=int, default=1024)
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--high_cpu_memory", action="store_true")
    parser.add_argument("--parameters_level", action="store_true")
    parser.add_argument("--compiler_transformer", action="store_true")
    parser.add_argument("--is_chorus", action="store_true")

    parser.add_argument("--id_num", type=int, default=1)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default='')
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--shuffle", action="store_true")

    args = parser.parse_args()

    insightface = FaceAnalysis(
        name="buffalo_l",
        root='pretrained_weights/insightface',
        providers=[('CUDAExecutionProvider', {'device_id': '0'})]
    )
    insightface.prepare(ctx_id=0, det_size=(640, 640))

    w2v2_model_path = f'pretrained_weights/chinese-wav2vec2-base'
    audio_processor = AudioProcessor(
        w2v2_model_path,
        device='cuda',
    )

    sd_pipeline = pipeline(
        task='speaker-diarization',
        model='iic/speech_campplus_speaker-diarization_common',
    )

    predictor = VideoInfer(
        model_path=args.model_path,
        quant_model=args.quant,
        world_size=args.gpu_num,
        is_offload=args.offload,
        offload_config=OffloadConfig(
            high_cpu_memory=args.high_cpu_memory,
            parameters_level=args.parameters_level,
            compiler_transformer=args.compiler_transformer,
        ),
        add_port=f'334{random.choice(range(10))}{random.choice(range(10))}'
    )
    print("finish pipeline init")

    process(
        args.image_path,
        args.audio_path,
        args.prompt_path,
        args.mask_path,
        args.output_path,
        max_size=args.max_size
    )
