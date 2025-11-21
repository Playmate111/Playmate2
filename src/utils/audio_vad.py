import os

import cv2
import numpy as np
import librosa
import scipy.io.wavfile as wf
import webrtcvad
import struct
from tqdm import tqdm


def voice_activate_indices_detect(
    audio_file,
    sample_rate=16000,
    sample_window=0.02,
    interval_frames_threshold=8,
    per_frame_len=640,
    reverse=False
):
    """
    超过6个连续音频窗都是语音，才加入结果中，否则一律认为静音

    :param audio_file:
    :return:
    """
    sample_overlap = sample_window

    v = webrtcvad.Vad(3)
    # rate, data = wf.read(audio_file)
    data, rate = librosa.load(audio_file, sr=sample_rate)
    audio_len = data.shape[0]
    data = data * 32767
    data = (data * np.where(np.abs(data) > 800, 1., 0.)).astype(np.int16)

    frame_radio = int(per_frame_len // (rate * sample_window))

    if len(data.shape) > 1:
        data = data[:, 0]
    # data = struct.pack("%dh" % len(data), *data)
    sample_start = 0
    detected_windows = np.array([])
    sample_window = int(rate * sample_window)
    sample_overlap = int(rate * sample_overlap)
    # 识别每个音频窗是否为语音
    for sample_start in tqdm(range(0, len(data) - sample_window, sample_window)):
    # while (sample_start < (len(data) - sample_window)):
    #     print(f'process {round(sample_start / len(data), 2) * 100} %')
        sample_end = sample_start + sample_window
        if sample_end >= len(data):
            sample_end = len(data) - 1
            sample_start = sample_end - sample_window - 1
        data_window = data[sample_start:sample_end]
        detected_windows = np.append(detected_windows, [sample_start, v.is_speech(data_window.tobytes(), rate)])
        # detected_windows = np.append(detected_windows, [sample_start, v.is_speech(data_window, rate)])
        # sample_start += sample_overlap

    if reverse:
        detected_windows = np.append(detected_windows, [len(data), 1])
    else:
        detected_windows = np.append(detected_windows, [len(data), 0])

    detected_windows = detected_windows.reshape(int(len(detected_windows) / 2), 2)

    indices = []
    section = []
    act_start, act_end = -1, -1
    for i, (_, flag) in enumerate(detected_windows):
        if reverse:
            if flag == 0:
                act_start = i if act_start == -1 else act_start
                act_end = i
            elif act_start != -1 and act_end - act_start >= interval_frames_threshold:
                section.append((int(detected_windows[act_start, 0]), int(detected_windows[i, 0])))
                indices += [j for j in range(act_start // frame_radio - 1, (act_end + 1) // frame_radio + 1)]
                act_start, act_end = -1, -1
            elif act_start != -1:
                act_start, act_end = -1, -1
        else:
            if flag == 1:
                act_start = i if act_start == -1 else act_start
                act_end = i
            elif act_start != -1 and act_end - act_start >= interval_frames_threshold:
                section.append((int(detected_windows[act_start, 0]), int(detected_windows[i, 0])))
                indices += [j for j in range(act_start // frame_radio, (act_end + 1) // frame_radio)]
                act_start, act_end = -1, -1
            elif act_start != -1:
                act_start, act_end = -1, -1

    if act_start != -1:
        section.append([int(detected_windows[act_start, 0]), int(detected_windows[act_end, 0])])

    return np.asarray(section), np.array(indices), audio_len
