# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午4:32
@Author  : Kend
@FileName: sed_cropper.py
@Software: PyCharm
@modifier:
SED 狗吠裁剪类 by tinyML
"""


import numpy as np
import tensorflow as tf
import librosa
from utils import extract_mfcc


class BarkCropper:
    def __init__(self, model_path, sr=16000, window_duration=0.2, step_ratio=0.5):
        self.model = tf.saved_model.load(model_path)
        self.sr = sr
        self.window_samples = int(window_duration * sr)
        self.step_samples = int(window_duration * step_ratio * sr)
        self.prob_threshold = 0.8  # 可调

    def _sliding_windows(self, audio):
        """生成滑窗片段"""
        windows = []
        starts = []
        for start in range(0, len(audio) - self.window_samples + 1, self.step_samples):
            end = start + self.window_samples
            seg = audio[start:end]
            windows.append(seg)
            starts.append(start)
        return np.array(windows), starts

    def crop_barks(self, audio_segment):
        """输入一段疑似狗吠音频，输出精确狗吠片段列表"""
        if len(audio_segment) < self.window_samples:
            return []  # 太短，无法处理

        windows, starts = self._sliding_windows(audio_segment)
        if len(windows) == 0:
            return []

        # 提取 MFCC
        mfccs = np.array([extract_mfcc(w, sr=self.sr) for w in windows])
        mfccs = np.expand_dims(mfccs, axis=-1)  # (N, 40, T, 1)

        # 推理
        probs = self.model(mfccs).numpy()[:, 1]  # 假设二分类，索引1=狗吠

        # 后处理：合并连续高概率窗口
        barks = []
        in_bark = False
        bark_start = 0

        for i, prob in enumerate(probs):
            if prob >= self.prob_threshold and not in_bark:
                in_bark = True
                bark_start = starts[i]
            elif prob < self.prob_threshold and in_bark:
                in_bark = False
                bark_end = starts[i] + self.window_samples
                barks.append(audio_segment[bark_start:bark_end])

        # 处理结尾
        if in_bark:
            bark_end = starts[-1] + self.window_samples
            barks.append(audio_segment[bark_start:bark_end])

        return barks