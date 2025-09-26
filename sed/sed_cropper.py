# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午4:32
@Author  : Kend
@FileName: sed_cropper.py
@Software: PyCharm
@modifier:
SED 狗吠裁剪类 by tinyML
"""


"""
狗吠 SED 精修模块（配合 bark_frontend 使用）
- 输入：降噪音频 + 候选片段列表
- 输出：精确狗吠片段（边界精修）
"""


import os
import numpy as np
import tensorflow as tf
import librosa


class BarkSEDRefiner:
    def __init__(
            self,
            tflite_model_path,
            sr=16000,
            dog_bark_class_ids=0,
            confidence_threshold=0.7
    ):
        """
        初始化精修器
        - dog_bark_class_ids: 狗吠类别的索引列表，如 [0, 1]
        """
        self.sr = sr
        self.dog_bark_class_ids = dog_bark_class_ids
        self.confidence_threshold = confidence_threshold  # 狗吠置信度阈值
        self.model_input_samples = int(0.2 * sr)  # 3200  # 窗口采样数
        self.step_samples = int(0.05 * sr)  # 800 (50ms)  # 滑窗步长

        # 加载 TFLite 模型
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def extract_mfcc(self, y):
        """提取 MFCC 特征（与训练一致）"""
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=40, n_fft=400, hop_length=160
        )
        mfcc = mfcc[np.newaxis, ..., np.newaxis]  # (1, 40, T, 1)
        return mfcc.astype(np.float32)


    def _infer_window(self, y_window):
        """对单个窗口进行推理"""
        try:
            features = self.extract_mfcc(y_window)
            input_shape = self.input_details[0]['shape']

            # 调整时间维度
            if features.shape[2] < input_shape[2]:
                pad = ((0, 0), (0, 0), (0, input_shape[2] - features.shape[2]), (0, 0))
                features = np.pad(features, pad, mode='constant')
            else:
                features = features[:, :, :input_shape[2], :]

            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            dog_prob = sum(output[0][i] for i in self.dog_bark_class_ids)
            print(f"  置信度: {dog_prob:.3f} (阈值: {self.confidence_threshold})")
            return dog_prob >= self.confidence_threshold, float(dog_prob)
        except Exception as e:
            print(f"  ⚠️ 推理失败: {e}")
            return False, 0.0

    def refine_bark_boundary(self, y_segment):
        """
        精修单个候选片段的狗吠边界
        返回: (local_start, local_end) 或 (0, 0) if no bark
        """
        seg_len = len(y_segment)

        # 情况1: 片段太短 (<0.2s)
        if seg_len < self.model_input_samples:
            padded = np.pad(y_segment, (0, self.model_input_samples - seg_len))
            is_bark, conf = self._infer_window(padded)
            if is_bark and conf >= self.confidence_threshold:
                return 0, seg_len
            else:
                return 0, 0

        # 情况2: 片段 >= 0.2s → 滑窗
        window_info = []  # [(start, end, prob), ...]

        # 正常滑窗（50ms 步长）
        num_windows = ((seg_len - self.model_input_samples) // self.step_samples) + 1
        for i in range(num_windows):
            start = i * self.step_samples
            end = start + self.model_input_samples
            if end > seg_len:
                break
            window = y_segment[start:end]
            is_bark, prob = self._infer_window(window)
            window_info.append((start, end, prob if is_bark else 0.0))

        # 关键：尾部对齐窗口（确保包含尾音）
        tail_start = seg_len - self.model_input_samples
        if tail_start >= 0 and (not window_info or tail_start > window_info[-1][0]):
            tail_window = y_segment[tail_start:tail_start + self.model_input_samples]
            is_bark, prob = self._infer_window(tail_window)
            window_info.append((tail_start, tail_start + self.model_input_samples, prob if is_bark else 0.0))

        if not window_info:
            return 0, 0

        # 合并连续阳性窗口
        threshold = self.confidence_threshold
        merged = []
        current_start = None
        current_end = None

        for start, end, prob in window_info:
            if prob >= threshold:
                if current_start is None:
                    current_start = start
                    current_end = end
                else:
                    current_end = max(current_end, end)
            else:
                if current_start is not None:
                    merged.append((current_start, current_end))
                    current_start = None
                    current_end = None

        if current_start is not None:
            merged.append((current_start, current_end))

        if not merged:
            return 0, 0

        # 返回最长的连续段
        longest = max(merged, key=lambda x: x[1] - x[0])
        return longest[0], longest[1]

    def refine_all_candidates(self, y_denoised, candidates):
        """
        精修所有候选片段
        返回: [(global_start, global_end), ...] 精确狗吠片段
        """
        precise_barks = []

        for i, (global_start, global_end) in enumerate(candidates):
            segment = y_denoised[global_start:global_end]
            if len(segment) == 0:
                continue

            local_start, local_end = self.refine_bark_boundary(segment)
            if local_end > local_start:  # 有效狗吠
                global_precise_start = global_start + local_start
                global_precise_end = global_start + local_end
                precise_barks.append((global_precise_start, global_precise_end))
                print(f"  ✅ 候选段 {i}: 精修后 [{local_start / self.sr:.2f}s - {local_end / self.sr:.2f}s]")
            else:
                print(f"  ❌ 候选段 {i}: 非狗吠")

        return precise_barks