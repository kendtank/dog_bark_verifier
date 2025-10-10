# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/28
@Author  : Kend
@FileName: test_bark_frontend_v17_savefig.py
@Description:
    - 保存可视化图像到 figures 文件夹
    - peaks 显示为全局降噪音频索引
"""

import csv
import numpy as np
from pathlib import Path
from bark_frontend_full_temp import preprocess_for_bark_detection

try:
    import soundfile as sf
except ImportError:
    import librosa
    sf = None

import matplotlib.pyplot as plt


def save_audio(y, sr, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if sf is not None:
        sf.write(str(filepath), y, sr)
    else:
        librosa.output.write_wav(str(filepath), y, sr)


def compute_similarity(y1, y2, sr=16000):
    if len(y1) < 200 or len(y2) < 200:
        return 0.0, 0.0

    import librosa
    from scipy.spatial.distance import cosine

    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13, n_fft=256, hop_length=64)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13, n_fft=256, hop_length=64)
    mfcc_sim = 1 - cosine(mfcc1.flatten(), mfcc2.flatten())

    mel1 = librosa.feature.melspectrogram(y=y1, sr=sr, n_mels=40, n_fft=256, hop_length=64, fmax=7800)
    mel2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=40, n_fft=256, hop_length=64, fmax=7800)
    mel1_db = librosa.power_to_db(mel1 + 1e-6)
    mel2_db = librosa.power_to_db(mel2 + 1e-6)
    mel_sim = 1 - cosine(mel1_db.flatten(), mel2_db.flatten())

    return mfcc_sim, mel_sim


def visualize_candidates(y, sr, candidates, peaks=None, title="Audio Candidates", save_path=None):
    times = np.arange(len(y)) / sr
    plt.figure(figsize=(12, 5))
    plt.plot(times, y, color='blue', label="Waveform")

    # 候选段
    for idx, (s, e) in enumerate(candidates):
        plt.axvspan(s / sr, e / sr, color='green', alpha=0.3, label='Candidate' if idx == 0 else None)

    # 全局峰值
    if peaks is not None and len(peaks) > 0:
        plt.scatter(peaks / sr, y[peaks], color='red', marker='x', label='Peaks')

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()


def test_single_audio(audio_path: Path, output_dir: Path, sr=16000):
    import librosa
    print(f"\n🔍 处理: {audio_path.name}")

    try:
        y_orig, _ = librosa.load(str(audio_path), sr=sr, mono=True)

        y_denoised, candidates, energy = preprocess_for_bark_detection(
            str(audio_path), sr=sr,
            absolute_threshold=1e-4
        )

        denoised_path = output_dir / f"denoised_{audio_path.stem}.wav"
        save_audio(y_denoised, sr, denoised_path)

        # === 计算全局 peaks ===
        from scipy.signal import find_peaks
        hop_length = 100
        global_peaks = []

        for start, end in candidates:
            segment_energy = energy[start//hop_length:end//hop_length]
            if len(segment_energy) < 2:
                continue
            peaks_local, _ = find_peaks(segment_energy,
                                        prominence=0.3 * np.max(segment_energy),
                                        distance=int(0.1 * sr / hop_length))
            peaks_global = start + peaks_local * hop_length
            global_peaks.extend(peaks_global)

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

            status = "✅" if mfcc_sim >= 0.99 and mel_sim >= 0.99 else "⚠️"
            print(f"  {status} 段 {idx:02d}: {mfcc_sim:.4f} / {mel_sim:.4f} "
                  f"({(1 - mfcc_sim) * 100:.2f}% / {(1 - mel_sim) * 100:.2f}%) "
                  f"[{start / sr:.2f}s - {end / sr:.2f}s]")

        # 保存图像
        visualize_candidates(y_denoised, sr, candidates,
                             peaks=np.array(global_peaks),
                             title=audio_path.name,
                             save_path=output_dir / "figures" / f"{audio_path.stem}.png")

        return results

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return []


def main():
    import os
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)

    TEST_DIR = Path("data/test_bark_samples")
    OUTPUT_DIR = Path("output/bark_frontend_test_full_temp")
    SR = 16000
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

    (OUTPUT_DIR / "segments").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

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
    print(f"🖼️ 可视化图像目录: {OUTPUT_DIR / 'figures'}")


if __name__ == '__main__':
    main()
