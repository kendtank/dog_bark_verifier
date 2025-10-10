# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/19 下午4:12
@Author  : Kend
@FileName: tiny_braking_improve.py
@Software: PyCharm
@modifier: Optimized for robustness (slower but safer)
"""


"""
可运行的音频增强脚本（支持文件或目录输入）
特点：
- 确保重采样到 target_sr
- 可选转为单声道（默认开启）
- 支持多种增强方法（按用户传入的方法顺序应用）
- 输出增强文件到指定目录，保持原始长度（除非某些增强改变长度）
使用示例：
python tiny_braking_improve.py --input path/to/dog.wav --output aug_out --methods noise,gain --num 5 --sr 16000 --mono 1
或对目录：
python tiny_braking_improve.py --input path/to/wavs_dir --output aug_out --methods noise,shift,pitch --num 3
"""

import os
import glob
import argparse
import random
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt


def load_audio(path, target_sr=16000, mono=True):
    """加载音频，保持原始通道或转单声道，并重采样到 target_sr"""
    y, orig_sr = librosa.load(path, sr=None, mono=False)

    if y.ndim > 1:
        if mono:
            y = librosa.to_mono(y)
        else:
            y = np.mean(y, axis=0)

    if orig_sr != target_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

    return y.astype(np.float32), target_sr


def denoise_audio_for_aug(y, sr=16000, low_freq=300, high_freq=8000, order=2):
    """轻量 Butterworth 带通滤波（用于增强）"""
    nyquist = sr / 2.0
    low = max(0.01, low_freq / nyquist)
    high = min(0.99, high_freq / nyquist)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, y)


class AudioAugmentor:
    def __init__(self, sr=16000, seed=None):
        self.sr = sr
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def add_noise(self, y, snr_db=20.0):
        rms = np.sqrt(np.mean(y ** 2)) + 1e-9
        # 静音保护：如果信号几乎为零，直接生成指定SNR的噪声
        if rms < 1e-6:
            noise_rms = 10 ** (-snr_db / 20.0)
            noise = np.random.randn(len(y)) * noise_rms
            return noise
        noise_rms = rms / (10 ** (snr_db / 20.0))
        noise = np.random.randn(len(y)) * noise_rms
        return y + noise

    def time_stretch(self, y, rate_range=(0.95, 1.05)):
        min_len = int(0.02 * self.sr)
        if len(y) < max(10, min_len):
            return y
        rate = float(np.random.uniform(*rate_range))
        try:
            y_st = librosa.effects.time_stretch(y, rate=rate)
            return y_st
        except Exception:
            return y

    def pitch_shift(self, y, n_steps_range=(-1, 1)):
        if len(y) < 2:
            return y
        steps = float(np.random.uniform(*n_steps_range))
        try:
            return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=steps)
        except Exception:
            return y

    def gain(self, y, db_range=(-6.0, 6.0)):
        db = float(np.random.uniform(*db_range))
        factor = 10 ** (db / 20.0)
        return y * factor

    def shift(self, y, shift_ms_range=(-50, 50)):
        """时间偏移：统一补零+截断，保持输出长度 = 输入长度"""
        shift_ms = int(np.random.uniform(*shift_ms_range))
        shift_samples = int(shift_ms / 1000.0 * self.sr)
        if shift_samples == 0:
            return y
        if shift_samples > 0:
            # 向右移：前补零
            y2 = np.concatenate([np.zeros(shift_samples, dtype=y.dtype), y])
        else:
            # 向左移：前截断，后补零
            shift_abs = abs(shift_samples)
            y2 = np.concatenate([y[shift_abs:], np.zeros(shift_abs, dtype=y.dtype)])
        # 统一截断到原始长度
        return y2[:len(y)]

    def band_filter(self, y, lowcut=300, highcut=8000):
        """使用 Butterworth 带通滤波（更鲁棒，无振铃）"""
        return denoise_audio_for_aug(y, sr=self.sr, low_freq=lowcut, high_freq=highcut)

    def apply(self, y, methods, params):
        y_out = y.copy()
        applied = []
        for m in methods:
            if m == "noise":
                snr = params.get('snr_db', 20.0)
                y_out = self.add_noise(y_out, snr_db=snr)
                applied.append(f"noise{snr}")
            elif m == "stretch":
                rng = params.get('stretch_range', (0.95, 1.05))
                y_out = self.time_stretch(y_out, rate_range=rng)
                applied.append(f"stretch{rng[0]:.2f}-{rng[1]:.2f}")
            elif m == "pitch":
                rng = params.get('pitch_steps', (-1, 1))
                y_out = self.pitch_shift(y_out, n_steps_range=rng)
                applied.append(f"pitch{rng[0]}-{rng[1]}")
            elif m == "gain":
                rng = params.get('gain_db', (-6, 6))
                y_out = self.gain(y_out, db_range=rng)
                applied.append(f"gain{rng[0]}-{rng[1]}")
            elif m == "shift":
                rng = params.get('shift_ms', (-50, 50))
                y_out = self.shift(y_out, shift_ms_range=rng)
                applied.append(f"shift{rng[0]}-{rng[1]}ms")
            elif m == "filter":
                low = params.get('filter_low', 300)
                high = params.get('filter_high', 8000)
                y_out = self.band_filter(y_out, lowcut=low, highcut=high)
                applied.append(f"filter{low}-{high}Hz")
            else:
                continue

        # 防止溢出：简单归一化
        max_abs = np.max(np.abs(y_out)) if y_out.size > 0 else 0.0
        if max_abs > 0.999:
            y_out = y_out / max_abs * 0.999

        return y_out, applied


def gather_input_files(input_path):
    exts = ['wav', 'mp3', 'flac', 'ogg', 'm4a']
    if os.path.isdir(input_path):
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_path, f"**/*.{ext}"), recursive=True))
        files = sorted(files)
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        raise ValueError(f"输入路径不存在: {input_path}")
    return files

"""
python tiny_braking_improve.py --input D:\work\datasets\tinyML\new_train\dog_braking\all --output D:\work\datasets\tinyML\new_train\dog_braking\all_aug --methods noise,stretch,pitch,gain,shift,filter --num 6

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入文件或目录')
    parser.add_argument('--output', default='aug_out', help='输出目录')
    parser.add_argument('--methods', default='noise,gain', help='逗号分隔的方法序列 (noise,stretch,pitch,gain,shift,filter)')
    parser.add_argument('--num', type=int, default=3, help='每个输入生成的增强样本数量')
    parser.add_argument('--sr', type=int, default=16000, help='目标采样率 (重采样)')
    parser.add_argument('--mono', type=int, choices=[0,1], default=1, help='是否转单声道 (1=是,0=否)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')

    # 可选参数
    parser.add_argument('--snr_db', type=float, default=20.0, help='添加噪声时的 SNR (dB)')
    parser.add_argument('--stretch_min', type=float, default=0.95)
    parser.add_argument('--stretch_max', type=float, default=1.05)
    parser.add_argument('--pitch_min', type=float, default=-1.0)
    parser.add_argument('--pitch_max', type=float, default=1.0)
    parser.add_argument('--gain_min', type=float, default=-6.0)
    parser.add_argument('--gain_max', type=float, default=6.0)
    parser.add_argument('--shift_min', type=int, default=-50)
    parser.add_argument('--shift_max', type=int, default=50)
    parser.add_argument('--filter_low', type=float, default=300.0)
    parser.add_argument('--filter_high', type=float, default=8000.0)

    args = parser.parse_args()

    files = gather_input_files(args.input)
    if len(files) == 0:
        print('没有找到音频文件')
        return

    os.makedirs(args.output, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    augmentor = AudioAugmentor(sr=args.sr, seed=args.seed)

    params = {
        'snr_db': args.snr_db,
        'stretch_range': (args.stretch_min, args.stretch_max),
        'pitch_steps': (args.pitch_min, args.pitch_max),
        'gain_db': (args.gain_min, args.gain_max),
        'shift_ms': (args.shift_min, args.shift_max),
        'filter_low': args.filter_low,
        'filter_high': args.filter_high,
    }

    total = 0

    for file in files:
        try:
            y, sr = load_audio(file, target_sr=args.sr, mono=bool(args.mono))
        except Exception as e:
            print(f"加载失败: {file} -> {e}")
            continue

        basename = os.path.splitext(os.path.basename(file))[0]
        # 总是保存原始文件（用于调试/分析）
        out_orig = os.path.join(args.output, f"{basename}_orig.wav")
        sf.write(out_orig, y, sr)
        total += 1

        # 只对足够长的音频做增强
        min_duration_sec = 0.25  # 250ms
        if len(y) >= min_duration_sec * args.sr:
            for i in range(args.num):
                y_aug, applied = augmentor.apply(y, methods, params)
                method_tag = "_".join(applied) if applied else "noop"
                method_tag = method_tag.replace(' ', '')[:80]
                out_path = os.path.join(args.output, f"{basename}_{method_tag}_{i+1}.wav")
                sf.write(out_path, y_aug, sr)
                total += 1
                print(f"Saved: {out_path}")
        else:
            print(f"跳过增强（音频太短: {len(y)/args.sr:.2f} 秒）: {file}")


if __name__ == '__main__':
    main()