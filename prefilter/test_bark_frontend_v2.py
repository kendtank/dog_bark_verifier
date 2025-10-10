# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/28 下午2:57
@Author  : Kend
@FileName: test_bark_frontend_v2.py
@Software: PyCharm
@modifier:
"""


"""
批量测试 bark_frontend 模块
- 输出每段狗吠候选的特征相似度
- 保存降噪音频 + 候选片段（供人工听测）
- 生成 CSV 报告
"""

import csv
import numpy as np
from pathlib import Path
from bark_frontend_v4 import preprocess_for_bark_detection

# === 音频保存工具 ===
try:
    import soundfile as sf
except ImportError:
    import librosa
    sf = None


def save_audio(y, sr, filepath: Path):
    """保存音频到文件"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if sf is not None:
        sf.write(str(filepath), y, sr)
    else:
        librosa.output.write_wav(str(filepath), y, sr)


# === 相似度计算 ===
def compute_similarity(y1, y2, sr=16000):
    """计算 MFCC / Log-Mel 相似度"""
    if len(y1) < 200 or len(y2) < 200:
        return 0.0, 0.0

    import librosa
    from scipy.spatial.distance import cosine

    # MFCC
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13, n_fft=256, hop_length=64)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13, n_fft=256, hop_length=64)
    mfcc_sim = 1 - cosine(mfcc1.flatten(), mfcc2.flatten())

    # Log-Mel
    mel1 = librosa.feature.melspectrogram(y=y1, sr=sr, n_mels=40, n_fft=256, hop_length=64, fmax=7800)
    mel2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=40, n_fft=256, hop_length=64, fmax=7800)
    mel1_db = librosa.power_to_db(mel1 + 1e-6)
    mel2_db = librosa.power_to_db(mel2 + 1e-6)
    mel_sim = 1 - cosine(mel1_db.flatten(), mel2_db.flatten())

    return mfcc_sim, mel_sim


# === 单条音频测试 ===
def test_single_audio(audio_path: Path, output_dir: Path, sr=16000):
    """测试单条音频，返回片段级结果列表"""
    import librosa
    print(f"\n🔍 处理: {audio_path.name}")

    try:
        # 加载音频
        y_orig, _ = librosa.load(str(audio_path), sr=sr, mono=True)

        # 预处理 + 候选段
        y_denoised, candidates = preprocess_for_bark_detection(str(audio_path), sr=sr,
                                                               multi_resolution=True,         # 启用多分辨率对比
                                                               absolute_threshold=1e-4,
                                                               diff_rel_threshold=0.5,
                                                               cooldown_ms=125
                                                               )

        # 保存降噪后音频
        denoised_path = output_dir / f"denoised_{audio_path.stem}.wav"
        save_audio(y_denoised, sr, denoised_path)

        results = []
        if not candidates:
            print("⚠️ 无候选段")
            return results

        for idx, (start, end) in enumerate(candidates):
            seg_path = output_dir / "segments" / f"{audio_path.stem}_seg{idx:03d}.wav"
            save_audio(y_denoised[start:end], sr, seg_path)

            mfcc_sim, mel_sim = compute_similarity(y_orig[start:end], y_denoised[start:end], sr=sr)

            result = {
                'filename': audio_path.name,
                'segment_id': idx,
                'start_sec': start / sr,
                'end_sec': end / sr,
                'duration_sec': (end - start) / sr,
                'mfcc_similarity': mfcc_sim,
                'mel_similarity': mel_sim,
                'mfcc_loss_pct': (1 - mfcc_sim) * 100,
                'mel_loss_pct': (1 - mel_sim) * 100,
                'denoised_audio': denoised_path.relative_to(output_dir),
                'segment_audio': seg_path.relative_to(output_dir)
            }
            results.append(result)

            # 控制台输出
            status = "✅" if mfcc_sim >= 0.99 and mel_sim >= 0.99 else "⚠️"
            print(f"  {status} 段 {idx:02d}: {mfcc_sim:.4f} / {mel_sim:.4f} "
                  f"({(1 - mfcc_sim) * 100:.2f}% / {(1 - mel_sim) * 100:.2f}%) "
                  f"[{start / sr:.2f}s - {end / sr:.2f}s]")

        return results

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return []


# === 主入口 ===
def main():
    import os
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)

    TEST_DIR = Path("data/test_bark_samples")
    OUTPUT_DIR = Path("output/bark_frontend_test_v4")
    SR = 16000
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

    (OUTPUT_DIR / "segments").mkdir(parents=True, exist_ok=True)

    audio_files = [f for f in TEST_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]
    print(f"📂 找到 {len(audio_files)} 个测试音频，开始批量处理...\n")

    all_results = []
    for audio_file in sorted(audio_files):
        results = test_single_audio(audio_file, OUTPUT_DIR, sr=SR)
        all_results.extend(results)

    if all_results:
        csv_path = OUTPUT_DIR / "similarity_report.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        total = len(all_results)
        passed = sum(1 for r in all_results if r['mfcc_similarity'] >= 0.99 and r['mel_similarity'] >= 0.99)
        print(f"\n📊 报告已保存: {csv_path}")
        print(f"🎯 总片段数: {total}, 达标数 (≤1% 损失): {passed} ({passed / total * 100:.1f}%)")
    else:
        print("⚠️ 没有有效结果")

    print(f"\n🎧 请人工听测片段目录: {OUTPUT_DIR / 'segments'}")
    print(f"🔊 降噪后完整音频目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
