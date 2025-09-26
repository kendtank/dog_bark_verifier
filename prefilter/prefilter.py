# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午4:32
@Author  : Kend
@FileName: prefilter.py
@Software: PyCharm
@modifier:
对输入的音频做预处理和前端初筛（能量/突发检测）
"""


"""
当前的策略——保守降噪（仅带通）+ 高召回初筛 + 模型兜底(取消增强，可能会引入失真，破坏声纹特征)
Google’s AudioSet / YAMNet：预处理仅做 resample + amplitude normalization，无降噪。
"""


import os
import numpy as np
import librosa
from scipy import signal


# 音频加载（重采样 + 单声道）
def load_audio(ori_audio, target_sr=16000):
    """加载音频：支持文件路径或 NumPy 数组"""
    if isinstance(ori_audio, str):
        if not os.path.exists(ori_audio):
            raise ValueError(f"文件不存在: {ori_audio}")
        y, sr = librosa.load(ori_audio, sr=None, mono=False)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    elif isinstance(ori_audio, np.ndarray):
        y = ori_audio.copy()
    else:
        raise ValueError("输入必须是文件路径或 NumPy 数组")

    # 转单声道
    if y.ndim > 1:
        y = librosa.to_mono(y)
    return y.astype(np.float32)




# 降噪去杂滤波（带通 + 谱减）
def denoise_audio(y, sr=16000, low_freq=200, high_freq=8000, order=2):
    """
    降噪策略：
    - 带通滤波：保留 200-8000Hz（狗吠主要频段）, 带通阶系数为2， 保留更多细节
    仅用带通（避免过度处理损伤狗吠， 且减少计算）
    mcu部署优化方向：
        滤波器系数固定：在 MCU 上，不能动态调用 butter 设计滤波器（会引入浮点运算和矩阵分解开销）。在PC上预先生成 b, a 系数，硬编码到 MCU 固件中。
                    或者改为 二阶 IIR 滤波器（Biquad Form II Direct），用 CMSIS-DSP 或 ARM IIR 库执行。
        运算方式优化：
            filtfilt 是前向+反向滤波（双向），在 MCU 上计算量太大。
            可以替换为单向 IIR 滤波器（虽然会有少量相位延迟，但不影响狗吠特征）。
        可选带宽调节：在 demo 阶段可以多测几个带宽，比如 300–7000Hz vs 200–8000Hz，看看 MFCC/Log-Mel 相似度的差异，找到最佳 trade-off。
    """
    # 1: 带通滤波（保留狗吠频段）
    nyquist = sr // 2
    
    # 确保频率在有效范围内
    low_freq = max(1, low_freq)  # 频率必须大于0
    high_freq = min(nyquist - 1, high_freq)  # 高频必须小于奈奎斯特频率
    
    # 归一化到0-1范围（scipy要求）
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # 确保频率在有效范围内
    low = max(0.01, low)  # 避免等于0
    high = min(0.99, high)  # 避免等于或超过1
    
    b, a = signal.butter(order, [low, high], btype='band')
    y_filtered = signal.filtfilt(b, a, y)

    return y_filtered



# 增强算法-尽量保证狗吠主体无损， 这里先不做处理，MCU实现开销大
def enhance_audio(y, sr=16000, energy_ratio_threshold=0.95):
    """
    增强原则：只处理非狗吠部分，狗吠主体保持原样
    通过能量频谱比判断是否为有效狗吠片段
    """
    # 计算全频段 vs 狗吠频段能量比
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    total_energy = np.sum(np.abs(fft) ** 2)
    band_energy = np.sum(np.abs(fft[(freqs >= 300) & (freqs <= 8000)]) ** 2)
    ratio = band_energy / (total_energy + 1e-8)

    # 如果已是高质量狗吠（ratio > threshold），直接返回
    if ratio >= energy_ratio_threshold:
        return y

    # ......
    # 这里取消：不做增强（因后续有 SED 模型兜底）
    return y



# 狗吠初筛函数（高召回率）
class HighRecallPrefilter:
    def __init__(self, sr=16000, min_duration=0.1, energy_threshold=0.005):
        self.sr = sr
        self.min_samples = int(min_duration * sr)
        self.energy_threshold = energy_threshold  # 低阈值 → 高召回

    def detect_candidates(self, y):
        """
        返回所有可能包含狗吠的候选片段（宁可多检，不可漏检）
        """
        frame_length = 400  # 25ms
        hop_length = 100  # 6.25ms（高时间分辨率）

        # 计算短时能量
        energy = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]

        # 低阈值检测（保证召回）
        events = []
        in_event = False
        start_frame = 0

        for i, e in enumerate(energy):
            if e > self.energy_threshold and not in_event:
                in_event = True
                start_frame = i
            elif e <= self.energy_threshold and in_event:
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

        return events



# 主流程：
def preprocess_for_bark_detection(ori_audio, sr=16000):
    y = load_audio(ori_audio, target_sr=sr)
    y_denoised = denoise_audio(y, sr=sr)
    y_enhanced = y_denoised

    prefilter = HighRecallPrefilter(sr=sr, energy_threshold=0.005)
    candidates = prefilter.detect_candidates(y_enhanced)  # 这个已经是 [(start, end), ...]

    # 不再提取 segments！直接返回位置
    return y_enhanced, candidates  # 返回处理后音频 + 候选位置


def compare_features_on_bark_segments(original, processed, candidates, sr=16000):
    """仅在检测到的狗吠候选段上计算特征相似度"""
    from scipy.spatial.distance import cosine

    total_mfcc_sim = 0.0
    total_mel_sim = 0.0
    valid_segments = 0

    for start, end in candidates:
        orig_seg = original[start:end]
        proc_seg = processed[start:end]

        # 跳过过短片段
        if len(orig_seg) < 200:  # ~12.5ms at 16kHz
            continue

        # 补零到相同长度（防边界误差）
        min_len = min(len(orig_seg), len(proc_seg))
        orig_seg = orig_seg[:min_len]
        proc_seg = proc_seg[:min_len]

        # 提取特征
        mfcc_orig = librosa.feature.mfcc(y=orig_seg, sr=sr, n_mfcc=13, n_fft=256, hop_length=64)
        mfcc_proc = librosa.feature.mfcc(y=proc_seg, sr=sr, n_mfcc=13, n_fft=256, hop_length=64)

        mel_orig = librosa.feature.melspectrogram(y=orig_seg, sr=sr, n_mels=40, n_fft=256, hop_length=64)
        mel_proc = librosa.feature.melspectrogram(y=proc_seg, sr=sr, n_mels=40, n_fft=256, hop_length=64)
        mel_orig_db = librosa.power_to_db(mel_orig + 1e-6)
        mel_proc_db = librosa.power_to_db(mel_proc + 1e-6)

        # 相似度
        mfcc_sim = 1 - cosine(mfcc_orig.flatten(), mfcc_proc.flatten())
        mel_sim = 1 - cosine(mel_orig_db.flatten(), mel_proc_db.flatten())

        total_mfcc_sim += mfcc_sim
        total_mel_sim += mel_sim
        valid_segments += 1

    if valid_segments == 0:
        print("未检测到有效狗吠段，无法评估特征保真度")
        return 0.0, 0.0

    avg_mfcc = total_mfcc_sim / valid_segments
    avg_mel = total_mel_sim / valid_segments

    print(f"狗吠主体段平均 MFCC 相似度: {avg_mfcc:.4f} ({(1 - avg_mfcc) * 100:.2f}% 损失)")
    print(f"狗吠主体段平均 Log-Mel 相似度: {avg_mel:.4f} ({(1 - avg_mel) * 100:.2f}% 损失)")

    return avg_mfcc, avg_mel



if __name__ == '__main__':
    # 使用 os.path.abspath 获取项目根目录的绝对路径
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("root_dir", root_dir)
    # 检查并加载音频 - 使用正确的路径
    ori_path = os.path.join(root_dir, "data", "ori_denoise_audio", "outdoor_braking_01.mp3")

    if not os.path.exists(ori_path):
        raise FileNotFoundError(f"测试文件不存在: {ori_path}")

    # 加载原始音频
    ori_y = load_audio(ori_path, target_sr=16000)

    y_processed, bark_segments_positions = preprocess_for_bark_detection(ori_path, sr=16000)

    # 现在 bark_segments_positions 是 [(start, end), ...]
    mfcc_sim, mel_sim = compare_features_on_bark_segments(
        ori_y, y_processed, bark_segments_positions, sr=16000
    )

"""
狗吠主体段平均 MFCC 相似度: 0.9965 (0.35% 损失)
狗吠主体段平均 Log-Mel 相似度: 0.9927 (0.73% 损失)
"""