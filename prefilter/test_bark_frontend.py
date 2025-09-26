# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/26 下午2:35
@Author  : Kend
@FileName: test_bark_frontend.py
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
from bark_frontend import preprocess_for_bark_detection

# 保存音频用（需安装 soundfile 或 librosa）
try:
    import soundfile as sf
except ImportError:
    import librosa
    sf = None  # 用 librosa.write_wav


def save_audio(y, sr, filepath):
    """保存音频（优先用 soundfile，回退到 librosa）"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if sf is not None:
        sf.write(str(filepath), y, sr)
    else:
        librosa.output.write_wav(str(filepath), y, sr)


def test_single_audio(audio_path, output_dir, sr=16000):
    """
    测试单条音频，返回片段级结果列表
    """
    print(f"\n🔍 处理: {audio_path.name}")
    try:
        # 加载并预处理
        y_orig = np.load(audio_path) if audio_path.suffix == '.npy' else None
        if y_orig is None:
            import librosa
            y_orig, _ = librosa.load(str(audio_path), sr=sr, mono=True)

        y_denoised, candidates = preprocess_for_bark_detection(str(audio_path), sr=sr)

        # 保存降噪后完整音频
        denoised_path = output_dir / f"denoised_{audio_path.stem}.wav"
        save_audio(y_denoised, sr, denoised_path)

        results = []
        if not candidates:
            print("  ⚠️ 无候选段")
            return results

        # 逐段验证 + 保存
        for idx, (start, end) in enumerate(candidates):
            # 保存候选片段
            seg_path = output_dir / "segments" / f"{audio_path.stem}_seg{idx:03d}.wav"
            save_audio(y_denoised[start:end], sr, seg_path)

            # 计算相似度（仅该段）
            orig_seg = y_orig[start:end]
            proc_seg = y_denoised[start:end]
            if len(orig_seg) < 200:
                mfcc_sim, mel_sim = 0.0, 0.0
            else:
                from scipy.spatial.distance import cosine
                import librosa
                # MFCC
                mfcc_orig = librosa.feature.mfcc(y=orig_seg, sr=sr, n_mfcc=13, n_fft=256, hop_length=64)
                mfcc_proc = librosa.feature.mfcc(y=proc_seg, sr=sr, n_mfcc=13, n_fft=256, hop_length=64)
                mfcc_sim = 1 - cosine(mfcc_orig.flatten(), mfcc_proc.flatten())
                # Log-Mel
                mel_orig = librosa.feature.melspectrogram(y=orig_seg, sr=sr, n_mels=40, n_fft=256, hop_length=64,
                                                          fmax=7800)
                mel_proc = librosa.feature.melspectrogram(y=proc_seg, sr=sr, n_mels=40, n_fft=256, hop_length=64,
                                                          fmax=7800)
                mel_orig_db = librosa.power_to_db(mel_orig + 1e-6)
                mel_proc_db = librosa.power_to_db(mel_proc + 1e-6)
                mel_sim = 1 - cosine(mel_orig_db.flatten(), mel_proc_db.flatten())

            # 记录结果
            results.append({
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
            })

            # 控制台输出
            status = "✅" if mfcc_sim >= 0.99 and mel_sim >= 0.99 else "⚠️"
            print(f"  {status} 段 {idx:02d}: {mfcc_sim:.4f} / {mel_sim:.4f} "
                  f"({(1 - mfcc_sim) * 100:.2f}% / {(1 - mel_sim) * 100:.2f}%) "
                  f"[{start / sr:.2f}s - {end / sr:.2f}s]")

        return results

    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return []


def main():
    import os
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)
    # === 配置 ===
    TEST_DIR = Path("data/test_bark_samples")  # 你的10个测试音频目录
    OUTPUT_DIR = Path("output/bark_frontend_test")  # 输出目录
    SR = 16000

    # 支持的音频格式
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

    # 创建输出目录
    (OUTPUT_DIR / "segments").mkdir(parents=True, exist_ok=True)

    # 批量测试
    all_results = []
    audio_files = [f for f in TEST_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]
    print(f"📁 找到 {len(audio_files)} 个测试音频，开始批量处理...\n")

    for audio_file in sorted(audio_files):
        results = test_single_audio(audio_file, OUTPUT_DIR, sr=SR)
        all_results.extend(results)

    # 保存 CSV 报告
    csv_path = OUTPUT_DIR / "similarity_report.csv"
    if all_results:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n📊 报告已保存: {csv_path}")

        # 统计达标率
        total = len(all_results)
        passed = sum(1 for r in all_results if r['mfcc_similarity'] >= 0.99 and r['mel_similarity'] >= 0.99)
        print(f"🎯 总片段数: {total}, 达标数 (≤1% 损失): {passed} ({passed / total * 100:.1f}%)")
    else:
        print("⚠️ 无有效结果")

    print(f"\n🎧 请人工听测片段目录: {OUTPUT_DIR / 'segments'}")
    print(f"🔊 降噪后完整音频目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

    """
    cd prefilter
    python test_bark_frontend.py
    """