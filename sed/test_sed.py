# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/26 下午4:00
@Author  : Kend
@FileName: test_sed.py
@Software: PyCharm
@modifier:
"""



"""
批量测试 BarkSEDRefiner
- 输入: 测试音频目录
- 输出: 对每个音频，若检测到狗吠，生成 precise_barks/{filename}/ 目录
"""

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(root_dir)
# 根目录加入到环境变量
sys.path.append(root_dir)

import shutil
from pathlib import Path
import numpy as np
from prefilter import preprocess_for_bark_detection
from sed_cropper import BarkSEDRefiner


def save_audio(y, sr, filepath):
    """保存音频（兼容新版 librosa）"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # 使用 soundfile（推荐）或新版 librosa.write
    try:
        import soundfile as sf
        sf.write(str(filepath), y, sr)
    except ImportError:
        # 回退到 librosa.write（librosa >=0.8.0）
        import librosa
        librosa.write(str(filepath), y, sr)


def test_single_audio(audio_path, output_base_dir, sed_refiner, sr=16000):
    """
    测试单条音频
    - 若检测到狗吠，保存精确片段到 output_base_dir / {stem} /
    - 否则不生成目录
    """
    print(f"\n🔍 处理: {audio_path.name}")

    try:
        # 1. 前端初筛
        y_denoised, candidates = preprocess_for_bark_detection(str(audio_path), sr=sr)

        if not candidates:
            print("  ⚠️ 无候选片段")
            return False

        # 2. SED 精修
        precise_barks = sed_refiner.refine_all_candidates(y_denoised, candidates)

        if not precise_barks:
            print("  ❌ 无有效狗吠")
            return False

        # 3. 保存精确片段
        output_dir = output_base_dir / audio_path.stem
        # 清理旧目录
        if output_dir.exists():
            shutil.rmtree(output_dir)

        for i, (start, end) in enumerate(precise_barks):
            bark_clip = y_denoised[start:end]
            clip_path = output_dir / f"bark_{i:03d}.wav"
            save_audio(bark_clip, sr, clip_path)

        print(f"  🎯 保存 {len(precise_barks)} 个精确狗吠片段到: {output_dir}")
        return True

    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False


def main():
    # === 配置 ===
    TEST_DIR = Path("data/test_bark_samples")  # 你的测试音频目录
    OUTPUT_DIR = Path("output/precise_barks")  # 输出目录
    TFLITE_MODEL = "model/tiny_cnn_bark.tflite"  # 你的 TFLite 模型路径

    # 根据你的训练数据修改！
    DOG_BARK_CLASS_IDS = [0]  # ← 必须正确设置！
    CONFIDENCE_THRESHOLD = 0.5

    SR = 16000
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

    # 检查模型
    if not os.path.exists(TFLITE_MODEL):
        raise FileNotFoundError(f"TFLite 模型不存在: {TFLITE_MODEL}")

    # 初始化 SED 精修器
    sed_refiner = BarkSEDRefiner(
        tflite_model_path=TFLITE_MODEL,
        sr=SR,
        dog_bark_class_ids=DOG_BARK_CLASS_IDS,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # 批量测试
    audio_files = [f for f in TEST_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]
    print(f"📁 找到 {len(audio_files)} 个测试音频\n")

    success_count = 0
    for audio_file in sorted(audio_files):
        has_bark = test_single_audio(audio_file, OUTPUT_DIR, sed_refiner, sr=SR)
        if has_bark:
            success_count += 1

    print(f"\n📊 总结: {success_count}/{len(audio_files)} 个音频检测到狗吠")
    if success_count > 0:
        print(f"🎧 精确狗吠片段已保存至: {OUTPUT_DIR}")
    else:
        print("⚠️ 未检测到任何狗吠")


if __name__ == '__main__':
    main()