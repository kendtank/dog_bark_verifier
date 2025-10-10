# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/28
@Author  : Kend
@FileName: bark_frontend_full.py
@Description:
    狗吠声纹前置初筛预处理
    - MCU优化 ZCR + 能量
    - 多分辨率检测
    - 自动峰值微调
    - 快速吼+快速叫自动切分
    - 可视化峰值事件 + 候选片段
    - 修复候选片段生成为空问题

整体架构
    预处理器采用多阶段处理流程，主要包括：
    音频加载和预处理
    带通滤波降噪
    能量和过零率计算
    候选区域检测
    峰值分析和片段细化
    片段合并和扩展
"""


import os
import numpy as np
from scipy.signal import butter, sosfilt, find_peaks
import librosa
import matplotlib.pyplot as plt


# ------------------------------
# 音频加载
# ------------------------------
def load_audio(ori_audio, target_sr=16000):
    if isinstance(ori_audio, str):
        if not os.path.exists(ori_audio):
            raise ValueError(f"文件不存在: {ori_audio}")
        y, sr = librosa.load(ori_audio, sr=None, mono=False)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    elif isinstance(ori_audio, np.ndarray):
        y = ori_audio.copy()
        sr = target_sr
    else:
        raise ValueError("输入必须是文件路径或 NumPy 数组")
    if y.ndim > 1:
        y = librosa.to_mono(y)
    return y.astype(np.float32), sr


# ------------------------------
# 简单带通降噪 避免损失狗吠主体特征
# ------------------------------
def denoise_audio(y, sr=16000, low_freq=300, high_freq=8000, order=2):
    nyquist = sr / 2.0
    low = max(0.01, low_freq / nyquist)
    high = min(0.99, high_freq / nyquist)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, y)


# ------------------------------
# 短时平方能量
# ------------------------------
def compute_short_time_energy(y, frame_length=400, hop_length=100):
    """  计算短时能量，帧长25ms，步长6.25ms """
    energy = []
    if len(y) < frame_length:
        return np.array([])
    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        energy.append(np.mean(frame * frame))   # 均方能量
    return np.array(energy)


# ------------------------------
# ZCR（过零次数）
# 过零率（Zero-Crossing Rate, ZCR）是音频信号处理中的一个重要特征：
# 定义：单位时间内信号通过零点的次数
# 作用：衡量信号的频率特性，高频信号通常有更高的过零率
# 实现：通过计算相邻样本符号变化的次数来统计
# ------------------------------
def compute_zcr_counts(y, frame_length=400, hop_length=100):
    """  # 计算过零率，衡量信号频率特性 """
    zc_counts = []
    if len(y) < frame_length:
        return np.array([])
    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        zc = np.sum((frame[:-1] * frame[1:]) < 0)  # 相邻样本符号变化次数
        zc_counts.append(int(zc))
    return np.array(zc_counts)

# ------------------------------
# 自动微调峰值参数
# ------------------------------
def auto_tune_peaks(energy, sr, hop_length=100):
    if len(energy) == 0:
        return 0.5, 1
    peak_prominence = max(0.05 * np.max(energy), 1e-7)
    peak_distance_samples = max(1, int(0.05 * sr / hop_length))  # 50ms最小间隔
    return peak_prominence, peak_distance_samples


# ------------------------------
# 高召回前置过滤器 算法核心部分， 采用基于能量的检测机制
# ------------------------------
class HighRecallPrefilterV17:

    def __init__(self, sr=16000, min_duration=0.05,
                 energy_ratio_high=2.5,  # 启动阈值提高
                 energy_ratio_low=0.7,  # 维持阈值降低
                 ema_alpha=0.01,
                 absolute_threshold=1e-6,
                 extend_head_ms=100,  # 关键：大幅增加上下文
                 extend_tail_ms=150,  # 关键：包含完整衰减
                 energy_valley_ratio=0.3,
                 fine_merge_gap_ms=80):  # 合并间隙稍增大
        self.sr = sr
        self.min_samples = int(min_duration * sr)
        self.energy_ratio_high = energy_ratio_high
        self.energy_ratio_low = energy_ratio_low
        self.ema_alpha = ema_alpha
        self.absolute_threshold = float(absolute_threshold)
        self.extend_head_samples = int(extend_head_ms / 1000 * sr)
        self.extend_tail_samples = int(extend_tail_ms / 1000 * sr)
        self.energy_valley_ratio = energy_valley_ratio
        self.fine_merge_gap_samples = int(fine_merge_gap_ms / 1000 * sr)

    def detect_candidates(self, y):
        frame_length, hop_length = 400, 100
        energy = compute_short_time_energy(y, frame_length, hop_length)
        events = []

        # =========================
        # Step1: 初步能量触发
        # =========================
        noise_floor = energy[0] if len(energy) > 0 else 1e-6
        in_event = False
        start_frame = 0

        for i, e in enumerate(energy):
            noise_floor = self.ema_alpha * e + (1 - self.ema_alpha) * noise_floor
            # 滞后阈值：高启动，低维持
            high_thresh = max(noise_floor * self.energy_ratio_high, self.absolute_threshold)
            low_thresh = max(noise_floor * self.energy_ratio_low, self.absolute_threshold * 0.5)

            if e > high_thresh and not in_event:
                in_event = True
                start_frame = i
            elif in_event and e < low_thresh:
                in_event = False
                start_sample = start_frame * hop_length
                end_sample = i * hop_length + frame_length
                events.append((start_sample, min(end_sample, len(y))))

        if in_event:  # 最后一段补全
            start_sample = start_frame * hop_length
            end_sample = len(y)
            events.append((start_sample, end_sample))

        # --------------------------
        # Step2: 峰值 + 包络修正，包络延伸（基于局部能量）
        # --------------------------
        refined_events = []
        for start, end in events:
            seg_start_frame = start // hop_length
            seg_end_frame = min(end // hop_length, len(energy))
            segment_energy = energy[seg_start_frame:seg_end_frame]

            if len(segment_energy) < 2:
                refined_events.append((start, end))
                continue

            # 使用局部最大值（更鲁棒）
            valley_threshold = np.max(segment_energy) * self.energy_valley_ratio
            peak_prominence, peak_distance_samples = auto_tune_peaks(segment_energy, self.sr, hop_length)
            peaks, _ = find_peaks(segment_energy, prominence=peak_prominence, distance=peak_distance_samples)

            if len(peaks) == 0:
                # 在 segment_energy 中找首次 >= valley_threshold 的位置（onset）
                s_idx = 0
                for i in range(len(segment_energy)):
                    if segment_energy[i] >= valley_threshold:
                        s_idx = i
                        break

                # 找最后一次 >= valley_threshold 的位置（offset）
                e_idx = len(segment_energy) - 1
                for i in range(len(segment_energy) - 1, -1, -1):
                    if segment_energy[i] >= valley_threshold:
                        e_idx = i
                        break

                # 转为全局样本点
                s_sample = max(0, (seg_start_frame + s_idx) * hop_length)
                e_sample = min(len(y), (seg_start_frame + e_idx + 1) * hop_length + frame_length)

                refined_events.append((s_sample, e_sample))
            else:
                for peak_local in peaks:
                    # 向左找 onset
                    onset_idx = 0
                    for i in range(peak_local, -1, -1):
                        if segment_energy[i] < valley_threshold:
                            onset_idx = i + 1
                            break

                    # 向右找 offset
                    offset_idx = len(segment_energy) - 1
                    for i in range(peak_local, len(segment_energy)):
                        if segment_energy[i] < valley_threshold:
                            # offset_idx = i - 1
                            offset_idx = max(0, i - 1)
                            break

                    s_frame = seg_start_frame + onset_idx
                    e_frame = seg_start_frame + offset_idx + 1
                    s_sample = max(0, s_frame * hop_length)
                    e_sample = min(len(y), e_frame * hop_length + frame_length)

                    # 保证最小长度
                    if e_sample - s_sample < self.min_samples:
                        center = (s_sample + e_sample) // 2
                        s_sample = max(0, center - self.min_samples // 2)
                        e_sample = min(len(y), center + self.min_samples // 2)

                    refined_events.append((s_sample, e_sample))

        # --------------------------
        # 合并邻近事件
        # --------------------------
        if len(refined_events) > 1:
            merged = [list(refined_events[0])]
            for s, e in refined_events[1:]:
                if s <= merged[-1][1] + self.fine_merge_gap_samples:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            refined_events = [(int(s), int(e)) for s, e in merged]

        # --------------------------
        # 扩展上下文（关键！）
        # --------------------------
        expanded = []
        for start, end in refined_events:
            s = max(0, start - self.extend_head_samples)
            e = min(len(y), end + self.extend_tail_samples)
            expanded.append((s, e))

        return expanded, energy


# ------------------------------
# 主流程
# ------------------------------
def preprocess_for_bark_detection(ori_audio, sr=16000, **kwargs):
    y, sr = load_audio(ori_audio, target_sr=sr)
    y_denoised = denoise_audio(y, sr=sr)
    prefilter = HighRecallPrefilterV17(sr=sr, **kwargs)
    candidates, energy = prefilter.detect_candidates(y_denoised)
    return y_denoised, candidates, energy

# ------------------------------
# 测试 + 可视化
# ------------------------------
def test_prefilter(audio_file, sr=16000, **kwargs):
    y_denoised, candidates, energy = preprocess_for_bark_detection(audio_file, sr, **kwargs)
    hop_length = 100
    times = np.arange(len(energy)) * (hop_length / sr)

    # 自动微调峰值
    peak_prominence, peak_distance_samples = auto_tune_peaks(energy, sr, hop_length)
    peaks, _ = find_peaks(energy, prominence=peak_prominence, distance=peak_distance_samples)

    plt.figure(figsize=(12, 6))
    plt.plot(times, energy, label="Energy", color="blue")
    plt.plot(times[peaks], energy[peaks], 'm*', label="Peaks")
    for s, e in candidates:
        plt.axvspan(s / sr, e / sr, color="green", alpha=0.3)
    plt.title(f"Prefilter V17 + Peaks: {os.path.basename(audio_file)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("检测到的候选片段（秒）:")
    for s, e in candidates:
        print(f"  [{s / sr:.2f} - {e / sr:.2f}]")

# ------------------------------
# 入口
# ------------------------------
if __name__ == "__main__":
    test_prefilter(
        r"D:\work\code\dog_bark_verifier\data\test_bark_samples\dog_in_home_003.mp3",
        sr=16000,
        absolute_threshold=1e-6,
        energy_valley_ratio=0.3,
        fine_merge_gap_ms=60,
        ema_alpha=0.01,
        energy_ratio_low=1.2,
        energy_ratio_high=3.0
    )
