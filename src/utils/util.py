import os
import torch
import subprocess
from src.utils.audio_vad import voice_activate_indices_detect


def get_file_list(data_dir, suffix="", max_num=None):
    file_list = []

    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(suffix):
                file_list.append(os.path.join(dirpath, filename))
                if max_num is not None and len(file_list) >= max_num:
                    break
        if max_num is not None and len(file_list) >= max_num:
            break

    try:
        file_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    except Exception as e:
        file_list = sorted(file_list)

    return file_list


def add_suffix_name(path, suffix):
    split_name = os.path.basename(path).split('.')
    name, format = '.'.join(split_name[:-1]), split_name[-1]
    return os.path.join(os.path.dirname(path), name + suffix + '.' + format)


def resample_audio(input_audio_file: str, output_audio_file: str, sample_rate: int):
    p = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-i", input_audio_file, "-ar", str(sample_rate), output_audio_file
    ])
    ret = p.wait()
    assert ret == 0, "Resample audio failed!"
    return output_audio_file


def add_audio_to_video(silent_video_path: str, audio_video_path: str, output_video_path: str):
    cmd = [
        'ffmpeg',
        '-y',
        '-i', f'"{silent_video_path}"',
        '-i', f'"{audio_video_path}"',
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-shortest',
        f'"{output_video_path}"'
    ]
    try:
        subprocess.run(' '.join(cmd), shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


def get_vad_mask(audio_path, fps=25, sample_rate=16000):
    vad_info = voice_activate_indices_detect(audio_path, per_frame_len=int(1 / fps * 16000), sample_rate=sample_rate)
    video_len = int(vad_info[2] / sample_rate * 25)
    indices = torch.from_numpy(vad_info[1])
    vad_mask = torch.zeros((video_len))
    vad_mask[indices] = 1

    return vad_mask
