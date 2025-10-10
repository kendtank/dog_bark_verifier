# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/26 下午5:54
@Author  : Kend
@FileName: build_esc50_dataset.py
@Software: PyCharm
@modifier:
"""


"""
从 ESC-50 构建狗吠检测训练数据集
- 正样本: ESC-50 中的 "dog" 类别
- 负样本: 指定的尖锐噪声类别
- 所有音频经过 bark_frontend 初筛处理
"""


import os
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from prefilter import preprocess_for_bark_detection


def save_audio(y, sr, filepath):
    """保存音频"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(filepath), y, sr)
    except ImportError:
        librosa.write(str(filepath), y, sr)


# 在 process_esc50_audio 函数后添加新的函数

def save_original_dog_audio(audio_path, output_dir, sr=16000):
    """
    保存原始狗音频到 origin_dog 文件夹
    """
    try:
        # 加载音频并重采样到 16kHz
        y, orig_sr = librosa.load(str(audio_path), sr=None, mono=True)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

        # 保存原始音频
        origin_dog_dir = output_dir / "origin_dog"
        origin_dog_dir.mkdir(exist_ok=True)

        output_file = origin_dog_dir / f"{audio_path.stem}.wav"
        save_audio(y, sr, output_file)
        return True

    except Exception as e:
        print(f"  ⚠️ 保存原始音频失败 {audio_path.name}: {e}")
        return False


def process_esc50_audio(audio_path, output_dir, class_name, sr=16000):
    """
    处理单个 ESC-50 音频：
    1. 重采样到 16kHz
    2. 前端初筛
    3. 保存候选片段到对应类别文件夹
    """
    try:
        # 加载音频（ESC-50 是 44.1kHz）
        y, orig_sr = librosa.load(str(audio_path), sr=None, mono=True)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

        y_denoised, candidates, _ = preprocess_for_bark_detection(
            y, sr=sr,
            multi_resolution=True,
            absolute_threshold=1e-4,
            diff_rel_threshold=0.5,
            cooldown_ms=125
        )

        saved_count = 0
        for i, (start, end) in enumerate(candidates):
            segment = y_denoised[start:end]
            if len(segment) < 3200:  # 小于 200ms 跳过
                continue

            # 保存片段（截取中间 200ms 用于训练）
            mid = len(segment) // 2
            start_idx = max(0, mid - 1600)
            end_idx = start_idx + 3200
            if end_idx > len(segment):
                end_idx = len(segment)
                start_idx = max(0, end_idx - 3200)

            clip = segment[start_idx:end_idx]
            if len(clip) == 3200:  # 确保正好 200ms
                output_file = output_dir / class_name / f"{audio_path.stem}_seg{i:03d}.wav"
                save_audio(clip, sr, output_file)
                saved_count += 1

        return saved_count

    except Exception as e:
        print(f"  ⚠️ 处理失败 {audio_path.name}: {e}")
        return 0


def main():
    # === 配置 ===
    ESC50_CSV = r"D:\work\datasets\ESC-50-master\meta\esc50.csv"  # ESC-50 元数据
    ESC50_AUDIO_DIR = r"D:\work\datasets\ESC-50-master\audio"  # ESC-50 音频目录
    OUTPUT_DATASET_DIR = "data/esc50_dog_dataset"  # 输出数据集目录

    # 负样本类别（尖锐噪声）
    NEGATIVE_CLASSES = [
        'keyboard_typing',
        'glass_breaking',
        'door_wood_knock',
        'metal',
        'siren',
        'bird',
        'clapping',
        'can_opening'
    ]

    SR = 16000
    MIN_SEGMENT_DURATION = 0.2  # 200ms

    # === 读取元数据 ===
    df = pd.read_csv(ESC50_CSV)
    print(f"📊 ESC-50 元数据加载完成，共 {len(df)} 条记录")

    # === 创建输出目录 ===
    output_dir = Path(OUTPUT_DATASET_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 处理正样本（dog）===
    print("\n🐕 处理正样本（dog）...")
    dog_files = df[df['category'] == 'dog']
    print(f"找到 {len(dog_files)} 个 dog 音频")

    dog_saved = 0
    origin_dog_saved = 0
    for _, row in dog_files.iterrows():
        audio_file = Path(ESC50_AUDIO_DIR) / row['filename']
        if audio_file.exists():
            # 保存原始音频
            if save_original_dog_audio(audio_file, output_dir, sr=SR):
                origin_dog_saved += 1

            # 处理并保存片段
            count = process_esc50_audio(audio_file, output_dir, 'dog', sr=SR)
            dog_saved += count
        else:
            print(f"  ❌ 文件不存在: {row['filename']}")

    print(f"✅ 保存 {origin_dog_saved} 个原始 dog 音频到 {output_dir}/origin_dog/")
    print(f"✅ 保存 {dog_saved} 个 dog 片段到 {output_dir}/dog/")


    # === 处理负样本（尖锐噪声）===
    print("\n🔊 处理负样本（尖锐噪声）...")
    negative_files = df[df['category'].isin(NEGATIVE_CLASSES)]
    print(f"找到 {len(negative_files)} 个负样本音频")

    neg_saved = 0
    for category in NEGATIVE_CLASSES:
        cat_files = negative_files[negative_files['category'] == category]
        print(f"  处理 {category}: {len(cat_files)} 个音频")

        for _, row in cat_files.iterrows():
            audio_file = Path(ESC50_AUDIO_DIR) / row['filename']
            if audio_file.exists():
                count = process_esc50_audio(audio_file, output_dir, category, sr=SR)
                neg_saved += count

    print(f"✅ 保存 {neg_saved} 个负样本片段")

    # === 统计 ===
    # 更新最终统计数据部分

    # === 统计 ===
    print(f"\n📈 最终数据集统计:")
    for class_dir in output_dir.iterdir():
        if class_dir.is_dir():
            wav_count = len(list(class_dir.glob("*.wav")))
            print(f"  {class_dir.name}: {wav_count} 个片段")

    print(f"\n📁 数据集已保存至: {output_dir}")


if __name__ == '__main__':
    main()

