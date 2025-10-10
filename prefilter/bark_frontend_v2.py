# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/28 下午2:33
@Author  : Kend $ GPT5
@FileName: bark_frontend_v2.py
@Software: PyCharm
@modifier:
"""

"""
修正版本（修复绝对阈值、多分辨率状态污染、稳健差分）
默认：
 - multi_resolution: False
 - use_log_for_diff: True
 - cooldown_ms: 125
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
    # 保证至少一帧
    if len(y) < frame_length:
        return np.array([])
    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        energy.append(np.mean(frame * frame))
    return np.array(energy)



# ------------------------------
# 高召回初筛器（修正版）
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
            cooldown_ms=125
    ):
        """
        参数说明（关键）：
        - absolute_threshold: 绝对能量阈值（兜底，当能量>此值时直接触发）
        - diff_rel_threshold: 相对/对数差分阈值（对数差分默认使用）
        - diff_abs_threshold: 绝对差分阈值（可选，优先级高于相对差分）
        - use_log_for_diff: 是否使用 log-energy 的差分（对幅度变化更稳健）
        - cooldown_ms: burst 冷却时间（ms），防止短时间内重复重置
        - multi_resolution: 是否启用多分辨率（OR 融合）
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
        self.noise_floor = 0.0  # persistent状态（流式场景可用）
        self.last_burst_frame = -100000
        self.multi_resolution = multi_resolution
        self.cooldown_ms = float(cooldown_ms)

    def _compute_energy_diff(self, energy):
        """差分计算：支持对数差分与普通差分（并做 3 点平滑）"""
        if len(energy) < 2:
            return np.zeros_like(energy)
        eps = 1e-12
        if self.use_log_for_diff:
            # 对数能量差：对不同音量尺度更稳健
            log_e = np.log(energy + eps)
            diff = np.diff(log_e, prepend=log_e[0])
        else:
            diff = np.diff(energy, prepend=energy[0])

        if len(diff) >= 3:
            smoothed = np.convolve(diff, [0.25, 0.5, 0.25], mode='same')
            return smoothed
        return diff

    def _detect_with_window(self, y, frame_length, hop_length, commit_noise=False):
        """
        在指定 window 上进行检测（不会默认修改 self.noise_floor，除非 commit_noise=True）
        返回：list of [start_sample, end_sample]
        """
        energy = compute_short_time_energy(y, frame_length, hop_length)
        if len(energy) == 0:
            return []

        energy_diff = self._compute_energy_diff(energy)
        median_energy = float(np.median(energy) if len(energy) > 0 else 0.0)

        # 转换 cooldown（帧数），frame_time_ms = hop_length / sr * 1000
        frame_time_ms = (hop_length / float(self.sr)) * 1000.0
        cooldown_frames = max(1, int(self.cooldown_ms / max(frame_time_ms, 1e-6)))

        MERGE_GAP_SAMPLES = 800

        events = []
        in_event = False
        start_frame = 0

        # 使用本地 noise_floor / last_burst_frame 避免污染 self.*
        noise_floor = float(self.noise_floor) if (self.noise_floor and self.noise_floor > 0.0) else float(energy[0])
        last_burst_frame = int(self.last_burst_frame)

        for i, e in enumerate(energy):
            current_ratio_low = self.energy_ratio_low
            current_ratio_high = self.energy_ratio_high

            # 快速噪声重置（local）
            if ((e > noise_floor * 5.0) or (e > self.absolute_threshold * 3.0)) and (i - last_burst_frame > cooldown_frames):
                noise_floor = e * 0.8
                current_ratio_low = 1.2
                current_ratio_high = 2.0
                last_burst_frame = i
            else:
                # EMA 本地更新
                if i == 0 and (noise_floor is None or noise_floor <= 0.0):
                    noise_floor = e
                else:
                    noise_floor = (self.ema_alpha * e + (1.0 - self.ema_alpha) * noise_floor)

            # 阈值（相对）
            relative_threshold_low = max(1e-12, noise_floor * current_ratio_low)
            relative_threshold_high = max(1e-12, noise_floor * current_ratio_high)

            # 稳健差分判断（优先使用绝对差分阈值）
            is_burst = False
            d = energy_diff[i]
            if self.use_log_for_diff:
                # 对数差分，直接比较 diff_rel_threshold（单位为 log-units）
                is_burst = (d > self.diff_rel_threshold)
                # 若提供绝对阈值，也可并用（可按需启用）
                if (self.diff_abs_threshold is not None) and (d > self.diff_abs_threshold):
                    is_burst = True
            else:
                # 原始差分：若用户给出绝对阈值优先使用
                if self.diff_abs_threshold is not None:
                    is_burst = (d > self.diff_abs_threshold)
                else:
                    # 用 median_energy 做兜底归一化，避免 noise_floor 很小时异常放大
                    denom = max(noise_floor, median_energy * 0.1, 1e-12)
                    is_burst = (d / denom) > self.diff_rel_threshold

            # 触发逻辑（修正后的 absolute 兜底 OR）
            triggered = False
            if e > self.absolute_threshold:
                # 绝对能量兜底：只要能量超过绝对阈值就触发（不再强制要求差分）
                triggered = True
            elif (e > relative_threshold_low) and is_burst:
                # 相对阈值 + 突发判定
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

        # 结尾处理
        if in_event:
            start_sample = start_frame * hop_length
            end_sample = len(y)
            if end_sample - start_sample >= self.min_samples:
                events.append((start_sample, end_sample))

        # 合并同一窗口内事件（以样本为单位）
        if not events:
            # 如果 commit_noise=True 也更新持久状态
            if commit_noise:
                self.noise_floor = noise_floor
                self.last_burst_frame = last_burst_frame
            return []

        merged = [list(events[0])]
        for start, end in events[1:]:
            if start <= merged[-1][1] + MERGE_GAP_SAMPLES:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        # 提交本地 noise 状态回 persistent（仅当 caller 要求）
        if commit_noise:
            self.noise_floor = noise_floor
            self.last_burst_frame = last_burst_frame

        return merged

    def detect_candidates(self, y):
        """
        detect_candidates:
         - 单分辨率 always run (window: 400/100)
         - multi_resolution True 时再 run 大窗（1024/256）
         两次运行不会互相污染 noise state（因为内部用本地变量）
        """
        events = []
        # 小窗（25 ms）
        small = self._detect_with_window(y, frame_length=400, hop_length=100, commit_noise=False)
        events.extend(small)

        # 可选大窗（64 ms）
        if self.multi_resolution:
            large = self._detect_with_window(y, frame_length=1024, hop_length=256, commit_noise=False)
            events.extend(large)

        if not events:
            return []

        # 融合两组事件（去重复、按时间排序并合并）
        events.sort(key=lambda x: x[0])
        merged = [list(events[0])]
        MERGE_GAP_SAMPLES = 800
        for start, end in events[1:]:
            if start <= merged[-1][1] + MERGE_GAP_SAMPLES:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        # 片段扩展
        expanded = []
        for start, end in merged:
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




import matplotlib.pyplot as plt


def test_prefilter(audio_file, sr=16000, multi_resolution=False, **kwargs):
    """
    测试函数：加载音频，运行预处理，并可视化能量+检测片段
    """
    # 1. 预处理 & 检测
    y_denoised, candidates = preprocess_for_bark_detection(
        audio_file, sr=sr, multi_resolution=multi_resolution, **kwargs
    )

    # 2. 计算能量序列（小窗，和 detect_candidates 对齐）
    frame_length, hop_length = 400, 100
    energy = compute_short_time_energy(y_denoised, frame_length, hop_length)

    # 构造时间轴
    times = np.arange(len(energy)) * (hop_length / sr)

    # 3. 简单 noise floor 曲线（EMA 模拟，不完全等于内部状态）
    noise_floor = []
    nf = energy[0]
    ema_alpha = kwargs.get("ema_alpha", 0.01)
    for e in energy:
        nf = ema_alpha * e + (1 - ema_alpha) * nf
        noise_floor.append(nf)
    noise_floor = np.array(noise_floor)

    # 4. 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(times, energy, label="Energy", color="blue")
    plt.plot(times, noise_floor, label="Noise floor (EMA)", color="red")
    plt.plot(times, noise_floor * kwargs.get("energy_ratio_low", 1.8),
             label="Relative threshold", color="orange", linestyle="--")

    # 标记检测到的片段
    for (s, e) in candidates:
        plt.axvspan(s / sr, e / sr, color="green", alpha=0.3)

    plt.title(f"Prefilter test: {os.path.basename(audio_file)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. 输出片段结果
    print("检测到的候选片段（秒）:")
    for (s, e) in candidates:
        print(f"  [{s/sr:.2f} - {e/sr:.2f}]")


if __name__ == "__main__":
    test_prefilter(
        r"D:\work\code\dog_bark_verifier\data\test_bark_samples\dog_in_home_003.mp3",
        multi_resolution=True,         # 启用多分辨率对比
        absolute_threshold=1e-4,
        diff_rel_threshold=0.5,
        cooldown_ms=125,
    )

