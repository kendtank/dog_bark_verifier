# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/28 下午2:33
@Author  : Kend $ Qwen3-MAX
@FileName: bark_frontend_v3.py
@Software: PyCharm
@modifier: MCU 友好版 ZCR 辅助分割（整数化 + 避免重复计算 + 智能合并）
"""

"""
MCU 优化要点：
1. ZCR 整数化：直接使用过零次数，避免浮点除法
2. 能量复用：在分割时复用 detect 阶段的能量序列
3. 智能合并：对过细片段进行二次合并
4. 所有阈值整数化：zc_threshold=60, energy_valley_ratio=0.3 → 整数比较
"""

import os
import numpy as np
from scipy.signal import butter, sosfilt


# ------------------------------
# 音频加载（文件路径或 numpy array）
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
# 简单带通降噪（SOS IIR）
# ------------------------------
def denoise_audio(y, sr=16000, low_freq=200, high_freq=8000, order=2):
    nyquist = sr / 2.0
    low = max(0.01, low_freq / nyquist)
    high = min(0.99, high_freq / nyquist)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, y)


# ------------------------------
# 短时平方能量（支持 window 参数）
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
# MCU 友好版过零率（返回过零次数，非归一化比率）
# ------------------------------
def compute_zcr_counts(y, frame_length=400, hop_length=100):
    """
    MCU 友好版 ZCR：返回过零次数（整数），避免浮点除法
    帧长 400 → 最大过零次数 ≈ 200
    """
    zc_counts = []
    if len(y) < frame_length:
        return np.array([])
    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        # 过零次数：相邻样本符号变化的次数
        zc = np.sum((frame[:-1] * frame[1:]) < 0)
        zc_counts.append(int(zc))  # 直接返回整数
    return np.array(zc_counts)


# ------------------------------
# 高召回初筛器（MCU 优化版）
# ------------------------------
class HighRecallPrefilter:
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
            zc_high_threshold=60,  # ZCR 高阈值（过零次数）
            zc_low_threshold=20,  # ZCR 低阈值（过零次数）
            energy_valley_ratio=0.3,  # 能量谷底比例（用于整数比较）
            fine_merge_gap_ms=60  # 过细片段合并间隔（ms）
    ):
        """
        MCU 优化参数：
        - zc_high_threshold: ZCR 高阈值（整数，过零次数）
        - zc_low_threshold: ZCR 低阈值（整数，过零次数）
        - energy_valley_ratio: 能量谷底比例（用于整数比较）
        - fine_merge_gap_ms: 过细片段合并间隔（ms）
        """
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
        """差分计算"""
        if len(energy) < 2:
            return np.zeros_like(energy)
        eps = 1e-12
        if self.use_log_for_diff:
            log_e = np.log(energy + eps)
            diff = np.diff(log_e, prepend=log_e[0])
        else:
            diff = np.diff(energy, prepend=energy[0])

        if len(diff) >= 3:
            smoothed = np.convolve(diff, [0.25, 0.5, 0.25], mode='same')
            return smoothed
        return diff

    def _detect_with_window_and_cache(self, y, frame_length, hop_length, commit_noise=False):
        """
        检测并缓存能量和 ZCR 序列，供后续分割使用
        返回: (events, energy_seq, zcr_seq, frame_info)
        """
        energy = compute_short_time_energy(y, frame_length, hop_length)
        if len(energy) == 0:
            return [], None, None, None

        # 缓存 ZCR（MCU 友好版）
        zcr_counts = compute_zcr_counts(y, frame_length, hop_length)

        # 缓存帧信息
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

        noise_floor = float(self.noise_floor) if (self.noise_floor and self.noise_floor > 0.0) else float(energy[0])
        last_burst_frame = int(self.last_burst_frame)

        for i, e in enumerate(energy):
            current_ratio_low = self.energy_ratio_low
            current_ratio_high = self.energy_ratio_high

            if ((e > noise_floor * 5.0) or (e > self.absolute_threshold * 3.0)) and (
                    i - last_burst_frame > cooldown_frames):
                noise_floor = e * 0.8
                current_ratio_low = 1.2
                current_ratio_high = 2.0
                last_burst_frame = i
            else:
                if i == 0 and (noise_floor is None or noise_floor <= 0.0):
                    noise_floor = e
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

        if not events:
            if commit_noise:
                self.noise_floor = noise_floor
                self.last_burst_frame = last_burst_frame
            return [], energy, zcr_counts, frame_info

        merged = [list(events[0])]
        for start, end in events[1:]:
            if start <= merged[-1][1] + MERGE_GAP_SAMPLES:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        if commit_noise:
            self.noise_floor = noise_floor
            self.last_burst_frame = last_burst_frame

        return merged, energy, zcr_counts, frame_info

    def _split_long_events_with_cached_data(self, y, events, energy, zcr_counts, frame_info):
        """
        使用缓存的能量和 ZCR 数据进行事件分割（避免重复计算）
        """
        if not self.zcr_split_enabled or energy is None or zcr_counts is None:
            return events

        if len(energy) != len(zcr_counts):
            return events

        split_events = []
        hop_length = frame_info['hop_length']

        for start, end in events:
            event_duration = end - start
            if event_duration < self.min_split_samples:
                split_events.append((start, end))
                continue

            # 找到事件对应的帧范围
            event_start_frame = None
            event_end_frame = None
            for i, frame_start in enumerate(frame_info['start_times']):
                if frame_start >= start:
                    event_start_frame = i
                    break
            for i in range(len(frame_info['end_times']) - 1, -1, -1):
                if frame_info['end_times'][i] <= end:
                    event_end_frame = i
                    break

            if event_start_frame is None or event_end_frame is None or event_start_frame > event_end_frame:
                split_events.append((start, end))
                continue

            # 在事件帧范围内找分割点
            split_frames = []

            # 1. 怒声→叫声转换：ZCR 从低到高突变
            for i in range(event_start_frame + 1, event_end_frame + 1):
                if (zcr_counts[i] > self.zc_high_threshold and
                        zcr_counts[i - 1] < self.zc_low_threshold):
                    split_frames.append(i)

            # 2. 连续吠叫间隔：能量谷底 + ZCR 低谷
            for i in range(event_start_frame + 1, event_end_frame):
                if (energy[i] < energy[i - 1] * self.energy_valley_ratio and
                        energy[i] < energy[i + 1] * self.energy_valley_ratio and
                        zcr_counts[i] < self.zc_low_threshold):
                    split_frames.append(i)

            split_frames = sorted(set(split_frames))

            if not split_frames:
                split_events.append((start, end))
                continue

            # 生成子事件
            last_split = event_start_frame
            for split in split_frames:
                sub_start = frame_info['start_times'][last_split]
                sub_end = frame_info['end_times'][split]
                if sub_end - sub_start >= self.min_samples:
                    split_events.append((sub_start, min(sub_end, end)))
                last_split = split

            # 添加最后一段
            last_start = frame_info['start_times'][last_split]
            if end - last_start >= self.min_samples:
                split_events.append((last_start, end))

        return split_events

    def _merge_fine_segments(self, events):
        """
        对过细片段进行二次合并（避免切得太碎）
        """
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
        """
        MCU 优化版 detect_candidates：
        1. 缓存能量和 ZCR 序列
        2. 复用缓存数据进行事件分割
        3. 对过细片段进行二次合并
        """
        all_events = []
        all_energy = []
        all_zcr = []
        all_frame_info = []

        # 小窗检测 + 缓存
        small_events, small_energy, small_zcr, small_frame_info = self._detect_with_window_and_cache(
            y, frame_length=400, hop_length=100, commit_noise=False
        )
        if small_events:
            all_events.extend(small_events)
            all_energy.append(small_energy)
            all_zcr.append(small_zcr)
            all_frame_info.append(small_frame_info)

        # 大窗检测 + 缓存
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

        # 融合多分辨率事件
        all_events.sort(key=lambda x: x[0])
        merged_events = [list(all_events[0])]
        MERGE_GAP_SAMPLES = 800
        for start, end in all_events[1:]:
            if start <= merged_events[-1][1] + MERGE_GAP_SAMPLES:
                merged_events[-1][1] = max(merged_events[-1][1], end)
            else:
                merged_events.append([start, end])

        # ====== ZCR 辅助分割（使用缓存数据）======
        if self.zcr_split_enabled and all_energy and all_zcr and all_frame_info:
            # 合并所有缓存数据（简单处理，实际可更精细）
            combined_energy = np.concatenate(all_energy) if len(all_energy) > 1 else all_energy[0]
            combined_zcr = np.concatenate(all_zcr) if len(all_zcr) > 1 else all_zcr[0]
            # 使用第一个 frame_info（简化处理）
            frame_info = all_frame_info[0]

            split_events = self._split_long_events_with_cached_data(
                y, merged_events, combined_energy, combined_zcr, frame_info
            )
        else:
            split_events = merged_events

        # ====== 过细片段二次合并 ======
        final_events = self._merge_fine_segments(split_events)

        # 片段扩展
        expanded = []
        for start, end in final_events:
            new_start = max(0, start - self.extend_head_samples)
            new_end = min(len(y), end + self.extend_tail_samples)
            expanded.append((int(new_start), int(new_end)))

        return expanded


# ------------------------------
# 主流程（供测试）
# ------------------------------
def preprocess_for_bark_detection(ori_audio, sr=16000, multi_resolution=False, **prefilter_kwargs):
    y = load_audio(ori_audio, target_sr=sr)
    y_denoised = denoise_audio(y, sr=sr)
    prefilter = HighRecallPrefilter(sr=sr, multi_resolution=multi_resolution, **prefilter_kwargs)
    candidates = prefilter.detect_candidates(y_denoised)
    return y_denoised, candidates


# ------------------------------
# 测试函数（保持不变）
# ------------------------------
import matplotlib.pyplot as plt


def test_prefilter(audio_file, sr=16000, multi_resolution=False, **kwargs):
    y_denoised, candidates = preprocess_for_bark_detection(
        audio_file, sr=sr, multi_resolution=multi_resolution, **kwargs
    )

    frame_length, hop_length = 400, 100
    energy = compute_short_time_energy(y_denoised, frame_length, hop_length)
    times = np.arange(len(energy)) * (hop_length / sr)

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

    plt.title(f"Prefilter test: {os.path.basename(audio_file)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("检测到的候选片段（秒）:")
    for (s, e) in candidates:
        print(f"  [{s / sr:.2f} - {e / sr:.2f}]")


if __name__ == "__main__":
    test_prefilter(
        r"D:\work\code\dog_bark_verifier\data\test_bark_samples\dog_in_home_003.mp3",
        multi_resolution=True,
        absolute_threshold=1e-4,
        diff_rel_threshold=0.5,
        cooldown_ms=125,
        zcr_split_enabled=True,
        min_split_duration=0.3,
        # MCU 优化参数
        zc_high_threshold=60,  # 帧长400，ZCR>60算高
        zc_low_threshold=20,  # ZCR<20算低
        energy_valley_ratio=0.3,  # 能量降到30%算谷底
        fine_merge_gap_ms=60  # 60ms内片段合并
    )