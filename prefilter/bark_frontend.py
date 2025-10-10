# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25
@Author  : Kend & Qwen3-MAX
@FileName: prefilter.py
@Description:
    狗吠声纹验证前端预处理模块（降噪 + 初筛）
    - 仅保留必要操作，确保 MFCC/Log-Mel 特征损失 ≤1%
    - 所有设计均面向 MCU 可移植性（无 librosa，无 sqrt，EMA 噪声估计等）
    设计原则： 初筛负责“不漏”，后端负责“不错”
"""


import os
import numpy as np
from scipy.signal import butter, sosfilt


# ================================
# 音频加载（重采样 + 单声道）
# ================================
def load_audio(ori_audio, target_sr=16000):
    """
    加载音频：支持文件路径或 NumPy 数组
    MCU 提示：实际部署时输入应为 float32[]，此函数仅用于测试
    """
    if isinstance(ori_audio, str):
        if not os.path.exists(ori_audio):
            raise ValueError(f"文件不存在: {ori_audio}")
        # 注意：此处仍用 librosa 仅用于测试，C 版输入为 raw array
        import librosa
        y, sr = librosa.load(ori_audio, sr=None, mono=False)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    elif isinstance(ori_audio, np.ndarray):
        y = ori_audio.copy()
    else:
        raise ValueError("输入必须是文件路径或 NumPy 数组")

    # 转单声道
    if y.ndim > 1:
        import librosa
        y = librosa.to_mono(y)
    return y.astype(np.float32)


# ================================
# 降噪：2阶 IIR 带通（SOS 形式）
# ================================
def denoise_audio(y, sr=16000, low_freq=200, high_freq=8000, order=2):
    """
    降噪策略：仅使用带通滤波（200–8000 Hz）
    - 使用 SOS（Second-Order Sections）形式，数值更稳定
    - MCU 可直接导出 sos 系数，用 arm_biquad_cascade_df1_f32 实现
    """
    nyquist = sr // 2
    low = max(0.01, low_freq / nyquist)
    high = min(0.99, high_freq / nyquist)

    # 生成 SOS（更适合 MCU 实现）
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, y)


# ================================
# 辅助：生成 C 可用的 SOS 系数（仅需运行一次）
# ================================
def export_sos_to_c(sr=16000, low_freq=200, high_freq=8000, order=2, name="bark_bandpass"):
    """
    生成 C 语言可用的 SOS 系数数组（用于 CMSIS-DSP）
    调用示例：export_sos_to_c()
    """
    nyquist = sr // 2
    low = max(0.01, low_freq / nyquist)
    high = min(0.99, high_freq / nyquist)
    sos = butter(order, [low, high], btype='band', output='sos')

    print(f"/* Auto-generated SOS coefficients for {name} */")
    print(f"#define {name.upper()}_NUM_SECTIONS {sos.shape[0]}")
    for i, section in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = section
        # CMSIS-DSP 要求系数顺序: [b0, b1, b2, -a1, -a2]
        print(f"static const float32_t {name}_sos{i}[5] = {{")
        print(f"    {b0:.8f}f, {b1:.8f}f, {b2:.8f}f, {-a1:.8f}f, {-a2:.8f}f")
        print(f"}};")


# ================================
# 能量计算：使用平方能量（无 sqrt）
# ================================
def compute_short_time_energy(y, frame_length=400, hop_length=100):
    """
    计算短时平方能量（mean(frame^2)）
    帧长 400 点（25ms），帧移 100 点（6.25ms）
    狗吠出现时，能量会突然升高（因为狗吠是突发性声音）
    MCU 提示：可直接用整数运算（Q15/Q31），无需 sqrt
    """
    energy = []
    for i in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[i:i + frame_length]
        energy.append(np.mean(frame * frame))  # 平方能量，无 sqrt
    return np.array(energy)


# ================================
# 高召回初筛（自适应阈值 + 事件合并）
# ================================
class HighRecallPrefilter:
    def __init__(
            self,
            sr=16000,
            min_duration=0.1,
            energy_ratio=3.0,
            ema_alpha=0.01,
            extend_head_ms=10,
            extend_tail_ms=50
    ):
        """
        初始化初筛器(宁可多检（把一些噪声当狗吠）, 绝不能漏检（把狗吠当噪声）)
        - energy_ratio: 能量阈值倍数（相对于背景噪声）这里设置的三倍
        - ema_alpha: EMA 噪声估计平滑因子（0.01 ≈ 时间常数 100 帧）
        - extend_head_ms: 向前扩展时间（ms），默认10ms（捕获起始瞬态）
        - extend_tail_ms: 向后扩展时间（ms），默认50ms（保留尾音）
        MCU 提示：ema_alpha 可转为定点移位（如 alpha = 1/128 → >>7）
        """
        self.sr = sr
        self.min_samples = int(min_duration * sr)  # 其实就是100ms
        self.energy_ratio = energy_ratio
        self.ema_alpha = ema_alpha
        self.extend_head_samples = int(extend_head_ms / 1000 * sr)
        self.extend_tail_samples = int(extend_tail_ms / 1000 * sr)
        self.noise_floor = 0.0  # 运行时状态，MCU 上需保留， 就是会一直存在内存中， 需要1kb

    def detect_candidates(self, y):
        """
        返回狗吠候选片段 [(start, end), ...]
        - 使用 EMA 估计背景噪声（替代 percentile）
        - 合并间隔 < 50ms 的事件（800 samples @16kHz）
        - 最终对每个片段进行头部/尾部扩展
        """
        energy = compute_short_time_energy(y)
        if len(energy) == 0:
            return []

        frame_length, hop_length = 400, 100
        MERGE_GAP_SAMPLES = 800  # 50ms @ 16kHz → 整型常量，MCU 友好

        events = []
        in_event = False
        start_frame = 0

        for i, e in enumerate(energy):
            """
            动态估计“背景噪声水平”（EMA）
            初始时，noise_floor ≈ 第一帧能量
            随着时间推移，它会缓慢跟踪背景噪声的平均水平
            狗吠出现时，能量突增，但 EMA 变化很慢（因为 alpha=0.01 很小）
            所以 noise_floor 仍接近背景噪声，不会被狗吠“带偏”
            EMA 只需 1 个变量比 percentile省mcu内存
            """
            # EMA 噪声估计（MCU 上只需一个状态变量）
            if i == 0:
                self.noise_floor = e
            else:
                self.noise_floor = (
                        self.ema_alpha * e + (1.0 - self.ema_alpha) * self.noise_floor
                )

            # 动态阈值检测 1e-6 是一个比数字噪声还低，但不会为零的安全下限
            threshold = max(1e-6, self.noise_floor * self.energy_ratio)


            # 事件检测（使用平方能量，阈值也是平方）
            """
            if energy > threshold → 认为“可能有狗吠”
            如果环境安静（noise_floor 小）→ 阈值低 → 容易触发（高召回）
            如果环境嘈杂（noise_floor 大）→ 阈值高 → 不会乱触发
            所以我们可以认为是自适应，这套参数，适用于室内/室外/安静/嘈杂场景
            
            生成候选片段:
            一旦能量 > 阈值 → 开始记录
            直到能量 ≤ 阈值 → 结束记录
            但要求片段 ≥ 100ms（min_duration=0.1），避免检测到瞬时脉冲噪声
            """

            if e > threshold and not in_event:
                in_event = True
                start_frame = i
            elif e <= threshold and in_event:
                in_event = False
                start_sample = start_frame * hop_length
                end_sample = i * hop_length + frame_length // 2
                if end_sample - start_sample >= self.min_samples:
                    events.append((start_sample, min(end_sample, len(y))))

        # 处理结尾
        if in_event:
            start_sample = start_frame * hop_length
            end_sample = len(y)
            if end_sample - start_sample >= self.min_samples:
                events.append((start_sample, end_sample))

        # 合并相邻事件（间隔 < 50ms）防碎片化
        # 如果两个狗吠间隔 < 50ms（比如连续吠叫），合并成一个片段,避免后端模型处理太多零碎片段
        if not events:
            return events

        merged = [list(events[0])]
        for start, end in events[1:]:
            if start <= merged[-1][1] + MERGE_GAP_SAMPLES:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        # 片段扩展， 避免丢失主题细节，防呆
        expanded_events = []
        for start, end in merged:
            # 向前扩展（但不超过音频开头）
            new_start = max(0, start - self.extend_head_samples)
            # 向后扩展（但不超过音频结尾）
            new_end = min(len(y), end + self.extend_tail_samples)
            expanded_events.append((new_start, new_end))

        return [(int(s), int(e)) for s, e in expanded_events]


# ================================
# 主流程
# ================================
def preprocess_for_bark_detection(
        ori_audio,
        sr=16000,
        low_freq=200,
        high_freq=8000,
        order=2,
        min_duration=0.1,
        energy_ratio=3.0,
        ema_alpha=0.01,
        extend_head_ms=10,
        extend_tail_ms=50
):
    """
    完整预处理 pipeline：
    1. 加载音频
    2. 降噪（SOS 带通）
    3. 初筛（EMA + 能量检测）
    返回：处理后音频, 候选位置列表 [(start, end), ...]
    """
    y = load_audio(ori_audio, target_sr=sr)
    y_denoised = denoise_audio(y, sr, low_freq, high_freq, order)
    prefilter = HighRecallPrefilter(sr, min_duration, energy_ratio, ema_alpha, extend_head_ms, extend_tail_ms)
    candidates = prefilter.detect_candidates(y_denoised)
    return y_denoised, candidates


# ================================
# 特征保真度验证（仅用于开发）
# ================================
def compare_features_on_bark_segments(original, processed, candidates, sr=16000):
    """仅在狗吠候选段上验证 MFCC/Log-Mel 相似度"""
    from scipy.spatial.distance import cosine
    import librosa

    total_mfcc_sim = 0.0
    total_mel_sim = 0.0
    valid_segments = 0

    for start, end in candidates:
        orig_seg = original[start:end]
        proc_seg = processed[start:end]
        if len(orig_seg) < 200:
            continue

        min_len = min(len(orig_seg), len(proc_seg))
        orig_seg = orig_seg[:min_len]
        proc_seg = proc_seg[:min_len]

        # 提取特征（仅用于验证）
        mfcc_orig = librosa.feature.mfcc(y=orig_seg, sr=sr, n_mfcc=13, n_fft=256, hop_length=64)
        mfcc_proc = librosa.feature.mfcc(y=proc_seg, sr=sr, n_mfcc=13, n_fft=256, hop_length=64)

        mel_orig = librosa.feature.melspectrogram(y=orig_seg, sr=sr, n_mels=40, n_fft=256, hop_length=64, fmax=7800)
        mel_proc = librosa.feature.melspectrogram(y=proc_seg, sr=sr, n_mels=40, n_fft=256, hop_length=64, fmax=7800)
        mel_orig_db = librosa.power_to_db(mel_orig + 1e-6)
        mel_proc_db = librosa.power_to_db(mel_proc + 1e-6)

        mfcc_sim = 1 - cosine(mfcc_orig.flatten(), mfcc_proc.flatten())
        mel_sim = 1 - cosine(mel_orig_db.flatten(), mel_proc_db.flatten())

        total_mfcc_sim += mfcc_sim
        total_mel_sim += mel_sim
        valid_segments += 1

    if valid_segments == 0:
        print("未检测到有效狗吠段")
        return 0.0, 0.0

    avg_mfcc = total_mfcc_sim / valid_segments
    avg_mel = total_mel_sim / valid_segments

    print(f"MFCC 相似度: {avg_mfcc:.4f} ({(1 - avg_mfcc) * 100:.2f}% 损失)")
    print(f"Log-Mel 相似度: {avg_mel:.4f} ({(1 - avg_mel) * 100:.2f}% 损失)")
    return avg_mfcc, avg_mel


# ================================
# 测试入口
# ================================
if __name__ == '__main__':
    # 生成 C 系数（只需运行一次）
    # export_sos_to_c()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ori_path = os.path.join(root_dir, "data", "ori_denoise_audio", "outdoor_braking_01.mp3")

    if not os.path.exists(ori_path):
        print(f"测试文件不存在: {ori_path}")
        exit(1)

    ori_y = load_audio(ori_path, target_sr=16000)
    y_processed, candidates = preprocess_for_bark_detection(ori_path, sr=16000)
    compare_features_on_bark_segments(ori_y, y_processed, candidates, sr=16000)

    """
    MFCC 相似度: 0.9981 (0.19% 损失)
    Log-Mel 相似度: 0.9953 (0.47% 损失)
    """