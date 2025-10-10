# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/28
@Author  : Kend $ Qwen3-MAX
@FileName: bark_frontend_v4.py
@Software: PyCharm
@modifier: MCU 友好版 V4（连续吠叫智能分割）
"""

import os
import numpy as np
from scipy.signal import butter, sosfilt

# ------------------------------
# 音频加载
# ------------------------------
def load_audio(ori_audio, target_sr=16000):
    if isinstance(ori_audio, str):
        if not os.path.exists(ori_audio):
            raise ValueError(f"文件不存在: {ori_audio}")
        import librosa
        y, sr = librosa.load(ori_audio, sr=None, mono=False)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    elif isinstance(ori_audio, np.ndarray):
        y = ori_audio.copy()
    else:
        raise ValueError("输入必须是文件路径或 NumPy 数组")
    if y.ndim > 1:
        import librosa
        y = librosa.to_mono(y)
    return y.astype(np.float32)

# ------------------------------
# 带通降噪（SOS IIR）
# ------------------------------
def denoise_audio(y, sr=16000, low_freq=200, high_freq=8000, order=2):
    nyquist = sr / 2.0
    low = max(0.01, low_freq / nyquist)
    high = min(0.99, high_freq / nyquist)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, y)

# ------------------------------
# 短时平方能量
# ------------------------------
def compute_short_time_energy(y, frame_length=400, hop_length=100):
    energy = []
    if len(y) < frame_length:
        return np.array([])
    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        energy.append(np.mean(frame * frame))
    return np.array(energy)

# ------------------------------
# MCU 友好版 ZCR（整数）
# ------------------------------
def compute_zcr_counts(y, frame_length=400, hop_length=100):
    zc_counts = []
    if len(y) < frame_length:
        return np.array([])
    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        zc = np.sum((frame[:-1] * frame[1:]) < 0)
        zc_counts.append(int(zc))
    return np.array(zc_counts)

# ------------------------------
# 高召回初筛器 V4
# ------------------------------
class HighRecallPrefilterV4:
    def __init__(
        self,
        sr=16000,
        min_duration=0.1,
        energy_ratio_high=3.0,
        energy_ratio_low=1.8,
        ema_alpha=0.01,
        absolute_threshold=1e-4,
        diff_rel_threshold=0.5,
        diff_abs_threshold=None,
        use_log_for_diff=True,
        extend_head_ms=10,
        extend_tail_ms=50,
        multi_resolution=False,
        cooldown_ms=125,
        zcr_split_enabled=True,
        min_split_duration=0.3,
        # MCU 优化参数
        zc_high_threshold=60,
        zc_low_threshold=20,
        energy_valley_ratio=0.35,
        fine_merge_gap_ms=60
    ):
        self.sr = sr
        self.min_samples = int(min_duration * sr)
        self.energy_ratio_high = energy_ratio_high
        self.energy_ratio_low = energy_ratio_low
        self.ema_alpha = ema_alpha
        self.absolute_threshold = float(absolute_threshold)
        self.diff_rel_threshold = diff_rel_threshold
        self.diff_abs_threshold = diff_abs_threshold
        self.use_log_for_diff = use_log_for_diff
        self.extend_head_samples = int(extend_head_ms / 1000 * sr)
        self.extend_tail_samples = int(extend_tail_ms / 1000 * sr)
        self.noise_floor = 0.0
        self.last_burst_frame = -100000
        self.multi_resolution = multi_resolution
        self.cooldown_ms = float(cooldown_ms)
        self.zcr_split_enabled = zcr_split_enabled
        self.min_split_samples = int(min_split_duration * sr)
        # MCU 优化参数
        self.zc_high_threshold = zc_high_threshold
        self.zc_low_threshold = zc_low_threshold
        self.energy_valley_ratio = energy_valley_ratio
        self.fine_merge_gap_samples = int(fine_merge_gap_ms / 1000 * sr)

    def _compute_energy_diff(self, energy):
        if len(energy) < 2:
            return np.zeros_like(energy)
        eps = 1e-12
        if self.use_log_for_diff:
            log_e = np.log(energy + eps)
            diff = np.diff(log_e, prepend=log_e[0])
        else:
            diff = np.diff(energy, prepend=energy[0])
        if len(diff) >= 3:
            return np.convolve(diff, [0.25, 0.5, 0.25], mode='same')
        return diff

    def _detect_with_window_and_cache(self, y, frame_length, hop_length, commit_noise=False):
        energy = compute_short_time_energy(y, frame_length, hop_length)
        if len(energy) == 0:
            return [], None, None, None
        zcr_counts = compute_zcr_counts(y, frame_length, hop_length)
        frame_info = {
            'frame_length': frame_length,
            'hop_length': hop_length,
            'start_times': np.arange(0, len(energy)) * hop_length,
            'end_times': np.arange(0, len(energy)) * hop_length + frame_length // 2
        }
        energy_diff = self._compute_energy_diff(energy)
        median_energy = float(np.median(energy) if len(energy) > 0 else 0.0)
        frame_time_ms = (hop_length / float(self.sr)) * 1000.0
        cooldown_frames = max(1, int(self.cooldown_ms / max(frame_time_ms, 1e-6)))
        MERGE_GAP_SAMPLES = 800

        events = []
        in_event = False
        start_frame = 0
        noise_floor = float(self.noise_floor) if self.noise_floor > 0 else float(energy[0])
        last_burst_frame = int(self.last_burst_frame)

        for i, e in enumerate(energy):
            current_ratio_low = self.energy_ratio_low
            current_ratio_high = self.energy_ratio_high
            if ((e > noise_floor * 5.0) or (e > self.absolute_threshold * 3.0)) and (i - last_burst_frame > cooldown_frames):
                noise_floor = e * 0.8
                current_ratio_low = 1.2
                current_ratio_high = 2.0
                last_burst_frame = i
            else:
                noise_floor = (self.ema_alpha * e + (1.0 - self.ema_alpha) * noise_floor)
            relative_threshold_low = max(1e-12, noise_floor * current_ratio_low)
            relative_threshold_high = max(1e-12, noise_floor * current_ratio_high)

            is_burst = False
            d = energy_diff[i]
            if self.use_log_for_diff:
                is_burst = (d > self.diff_rel_threshold)
                if (self.diff_abs_threshold is not None) and (d > self.diff_abs_threshold):
                    is_burst = True
            else:
                if self.diff_abs_threshold is not None:
                    is_burst = (d > self.diff_abs_threshold)
                else:
                    denom = max(noise_floor, median_energy * 0.1, 1e-12)
                    is_burst = (d / denom) > self.diff_rel_threshold

            triggered = False
            if e > self.absolute_threshold:
                triggered = True
            elif (e > relative_threshold_low) and is_burst:
                triggered = True

            if triggered and not in_event:
                in_event = True
                start_frame = i
            elif (e <= relative_threshold_high) and in_event:
                in_event = False
                start_sample = start_frame * hop_length
                end_sample = i * hop_length + frame_length // 2
                if end_sample - start_sample >= self.min_samples:
                    events.append((start_sample, min(end_sample, len(y))))
        if in_event:
            start_sample = start_frame * hop_length
            end_sample = len(y)
            if end_sample - start_sample >= self.min_samples:
                events.append((start_sample, end_sample))

        # 合并相邻事件
        if events:
            merged = [list(events[0])]
            for start, end in events[1:]:
                if start <= merged[-1][1] + MERGE_GAP_SAMPLES:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            events = merged

        if commit_noise:
            self.noise_floor = noise_floor
            self.last_burst_frame = last_burst_frame

        return events, energy, zcr_counts, frame_info

    def _split_long_events_with_valley(self, y, events, energy, zcr_counts, frame_info):
        """V4 核心：内部能量谷 + ZCR 联合二次切分"""
        if not events or energy is None or zcr_counts is None:
            return events

        split_events = []
        hop_length = frame_info['hop_length']
        start_times = frame_info['start_times']
        end_times = frame_info['end_times']

        for start, end in events:
            event_start_frame = np.searchsorted(start_times, start)
            event_end_frame = np.searchsorted(end_times, end) - 1
            event_frames = np.arange(event_start_frame, event_end_frame + 1)
            sub_splits = []

            for i in event_frames[1:-1]:
                # 能量谷 + ZCR 低点判定
                if (energy[i] < energy[i-1]*self.energy_valley_ratio and
                    energy[i] < energy[i+1]*self.energy_valley_ratio and
                    zcr_counts[i] < self.zc_low_threshold):
                    sub_splits.append(i)
            if not sub_splits:
                split_events.append((start, end))
                continue

            last_frame = event_start_frame
            for split in sub_splits:
                s_sample = start_times[last_frame]
                e_sample = end_times[split]
                if e_sample - s_sample >= self.min_samples:
                    split_events.append((s_sample, min(e_sample, end)))
                last_frame = split
            # 最后一段
            s_sample = start_times[last_frame]
            if end - s_sample >= self.min_samples:
                split_events.append((s_sample, end))

        return split_events

    def _merge_fine_segments(self, events):
        if len(events) <= 1:
            return events
        merged = [list(events[0])]
        for start, end in events[1:]:
            if start <= merged[-1][1] + self.fine_merge_gap_samples:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(int(s), int(e)) for s, e in merged]

    def detect_candidates(self, y):
        all_events, all_energy, all_zcr, all_frame_info = [], [], [], []

        # 小窗
        small_events, small_energy, small_zcr, small_frame_info = self._detect_with_window_and_cache(
            y, frame_length=400, hop_length=100, commit_noise=False
        )
        if small_events:
            all_events.extend(small_events)
            all_energy.append(small_energy)
            all_zcr.append(small_zcr)
            all_frame_info.append(small_frame_info)

        # 大窗
        if self.multi_resolution:
            large_events, large_energy, large_zcr, large_frame_info = self._detect_with_window_and_cache(
                y, frame_length=1024, hop_length=256, commit_noise=False
            )
            if large_events:
                all_events.extend(large_events)
                all_energy.append(large_energy)
                all_zcr.append(large_zcr)
                all_frame_info.append(large_frame_info)

        if not all_events:
            return []

        # 合并多分辨率事件
        all_events.sort(key=lambda x: x[0])
        merged_events = [list(all_events[0])]
        MERGE_GAP_SAMPLES = 800
        for start, end in all_events[1:]:
            if start <= merged_events[-1][1] + MERGE_GAP_SAMPLES:
                merged_events[-1][1] = max(merged_events[-1][1], end)
            else:
                merged_events.append([start, end])

        # 内部二次切分
        if self.zcr_split_enabled and all_energy and all_zcr and all_frame_info:
            combined_energy = np.concatenate(all_energy) if len(all_energy) > 1 else all_energy[0]
            combined_zcr = np.concatenate(all_zcr) if len(all_zcr) > 1 else all_zcr[0]
            frame_info = all_frame_info[0]
            split_events = self._split_long_events_with_valley(
                y, merged_events, combined_energy, combined_zcr, frame_info
            )
        else:
            split_events = merged_events

        final_events = self._merge_fine_segments(split_events)

        # 扩展
        expanded = []
        for start, end in final_events:
            new_start = max(0, start - self.extend_head_samples)
            new_end = min(len(y), end + self.extend_tail_samples)
            expanded.append((int(new_start), int(new_end)))

        return expanded

# ------------------------------
# 主流程
# ------------------------------
def preprocess_for_bark_detection(ori_audio, sr=16000, multi_resolution=False, **prefilter_kwargs):
    y = load_audio(ori_audio, target_sr=sr)
    y_denoised = denoise_audio(y, sr=sr)
    prefilter = HighRecallPrefilterV4(sr=sr, multi_resolution=multi_resolution, **prefilter_kwargs)
    candidates = prefilter.detect_candidates(y_denoised)
    return y_denoised, candidates

# ------------------------------
# 测试函数（可视化）
# ------------------------------
import matplotlib.pyplot as plt

def test_prefilter(audio_file, sr=16000, multi_resolution=False, **kwargs):
    y_denoised, candidates = preprocess_for_bark_detection(
        audio_file, sr=sr, multi_resolution=multi_resolution, **kwargs
    )

    frame_length, hop_length = 400, 100
    energy = compute_short_time_energy(y_denoised, frame_length, hop_length)
    times = np.arange(len(energy)) * (hop_length / sr)

    # EMA noise floor
    noise_floor = []
    nf = energy[0]
    ema_alpha = kwargs.get("ema_alpha", 0.01)
    for e in energy:
        nf = ema_alpha * e + (1 - ema_alpha) * nf
        noise_floor.append(nf)
    noise_floor = np.array(noise_floor)

    plt.figure(figsize=(12, 6))
    plt.plot(times, energy, label="Energy", color="blue")
    plt.plot(times, noise_floor, label="Noise floor (EMA)", color="red")
    plt.plot(times, noise_floor * kwargs.get("energy_ratio_low", 1.8),
             label="Relative threshold", color="orange", linestyle="--")
    for (s, e) in candidates:
        plt.axvspan(s / sr, e / sr, color="green", alpha=0.3)
    plt.title(f"Prefilter V4 test: {os.path.basename(audio_file)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("检测到的候选片段（秒）:")
    for (s, e) in candidates:
        print(f"  [{s / sr:.2f} - {e / sr:.2f}]")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    test_prefilter(
        r"D:\work\code\dog_bark_verifier\data\test_bark_samples\dog_in_home_003.mp3",
        multi_resolution=True,
        absolute_threshold=1e-4,
        diff_rel_threshold=0.5,
        cooldown_ms=125,
        zcr_split_enabled=True,
        min_split_duration=0.3,
        zc_high_threshold=60,
        zc_low_threshold=20,
        energy_valley_ratio=0.35,
        fine_merge_gap_ms=60
    )
