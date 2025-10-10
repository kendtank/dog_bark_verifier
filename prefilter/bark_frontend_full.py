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
                 energy_ratio_high=3.0, energy_ratio_low=1.2,
                 ema_alpha=0.01, absolute_threshold=1e-6,
                 diff_rel_threshold=0.3, extend_head_ms=20,
                 extend_tail_ms=60, multi_resolution=False,
                 cooldown_ms=50, zcr_split_enabled=True,
                 min_split_duration=0.03, zc_high_threshold=60,
                 zc_low_threshold=20, energy_valley_ratio=0.3,
                 fine_merge_gap_ms=60):
        self.sr = sr
        self.min_samples = int(min_duration * sr)
        self.energy_ratio_high = energy_ratio_high
        self.energy_ratio_low = energy_ratio_low
        self.ema_alpha = ema_alpha
        self.absolute_threshold = float(absolute_threshold)
        self.diff_rel_threshold = diff_rel_threshold
        self.extend_head_samples = int(extend_head_ms / 1000 * sr)
        self.extend_tail_samples = int(extend_tail_ms / 1000 * sr)
        self.multi_resolution = multi_resolution
        self.cooldown_ms = float(cooldown_ms)
        self.zcr_split_enabled = zcr_split_enabled
        self.min_split_samples = int(min_split_duration * sr)
        self.zc_high_threshold = zc_high_threshold
        self.zc_low_threshold = zc_low_threshold
        self.energy_valley_ratio = energy_valley_ratio
        self.fine_merge_gap_samples = int(fine_merge_gap_ms / 1000 * sr)

    # ------------------------------
    # 核心检测
    # ------------------------------
    def detect_candidates(self, y):
        frame_length, hop_length = 400, 100
        energy = compute_short_time_energy(y, frame_length, hop_length)
        # 目前方案只基于能量序列
        # zcr = compute_zcr_counts(y, frame_length, hop_length)
        events = []

        noise_floor = energy[0] if len(energy) > 0 else 1e-6
        in_event = False
        start_frame = 0
        cooldown_frames = max(1, int(self.cooldown_ms / (hop_length / self.sr * 1000)))

        for i, e in enumerate(energy):
            noise_floor = self.ema_alpha * e + (1 - self.ema_alpha) * noise_floor
            triggered = (e > max(noise_floor * self.energy_ratio_low, self.absolute_threshold))
            if triggered and not in_event:
                in_event = True
                start_frame = i
            elif (not triggered) and in_event:
                in_event = False
                start_sample = start_frame * hop_length
                end_sample = i * hop_length + frame_length
                events.append((start_sample, min(end_sample, len(y))))

        if in_event:
            start_sample = start_frame * hop_length
            end_sample = len(y)
            events.append((start_sample, end_sample))

        # --------------------------
        # 快速吼+快速叫分割 + 包络延伸（V18增强版）
        # --------------------------
        refined_events = []
        for start, end in events:
            seg_start_frame = start // hop_length
            seg_end_frame = min(end // hop_length, len(energy))
            segment_energy = energy[seg_start_frame:seg_end_frame]

            if len(segment_energy) < 2:
                refined_events.append((start, end))
                continue

            # 使用全局能量计算 valley_threshold 更鲁棒
            valley_threshold = np.max(energy) * self.energy_valley_ratio
            peak_prominence, peak_distance_samples = auto_tune_peaks(segment_energy, self.sr, hop_length)
            peaks, _ = find_peaks(segment_energy, prominence=peak_prominence, distance=peak_distance_samples)

            if len(peaks) == 0:
                # 无峰值：尝试用整个段的包络延伸
                s_frame, e_frame = seg_start_frame, seg_end_frame
                # 向左找 onset
                for i in range(len(segment_energy)):
                    if segment_energy[i] >= valley_threshold:
                        s_frame = seg_start_frame + i
                        break
                # 向右找 offset
                for i in range(len(segment_energy) - 1, -1, -1):
                    if segment_energy[i] >= valley_threshold:
                        e_frame = seg_start_frame + i + 1
                        break
                s_sample = max(0, s_frame * hop_length)
                e_sample = min(len(y), e_frame * hop_length + frame_length)
                refined_events.append((s_sample, e_sample))
            else:
                for peak_local in peaks:
                    # 以当前峰值为中心，向左右延伸至能量谷底
                    # 向左找 onset
                    onset_idx = peak_local
                    for i in range(peak_local, -1, -1):
                        if segment_energy[i] < valley_threshold:
                            onset_idx = i + 1
                            break
                    else:
                        onset_idx = 0

                    # 向右找 offset
                    offset_idx = peak_local
                    for i in range(peak_local, len(segment_energy)):
                        if segment_energy[i] < valley_threshold:
                            offset_idx = i
                            break
                    else:
                        offset_idx = len(segment_energy) - 1

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
        # 合并过短片段
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
        # 扩展片段
        # --------------------------
        expanded = []
        for start, end in refined_events:
            expanded.append((max(0, start - self.extend_head_samples),
                             min(len(y), end + self.extend_tail_samples)))

        return expanded, energy

# ------------------------------
# 主流程
# ------------------------------
def preprocess_for_bark_detection(ori_audio, sr=16000, multi_resolution=False, **kwargs):
    y, sr = load_audio(ori_audio, target_sr=sr)
    y_denoised = denoise_audio(y, sr=sr)
    prefilter = HighRecallPrefilterV17(sr=sr, multi_resolution=multi_resolution, **kwargs)
    candidates, energy = prefilter.detect_candidates(y_denoised)
    return y_denoised, candidates, energy

# ------------------------------
# 测试 + 可视化
# ------------------------------
def test_prefilter(audio_file, sr=16000, multi_resolution=False, **kwargs):
    y_denoised, candidates, energy = preprocess_for_bark_detection(audio_file, sr, multi_resolution, **kwargs)
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
        multi_resolution=True,
        absolute_threshold=1e-6,
        diff_rel_threshold=0.3,
        cooldown_ms=50,
        zcr_split_enabled=True,
        min_split_duration=0.03,
        zc_high_threshold=60,
        zc_low_threshold=20,
        energy_valley_ratio=0.3,
        fine_merge_gap_ms=60,
        ema_alpha=0.01,
        energy_ratio_low=1.2,
        energy_ratio_high=3.0
    )
