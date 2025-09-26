# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午4:36
@Author  : Kend
@FileName: demo.py
@Software: PyCharm
@modifier:
主演示脚本
"""

# demo.py
import librosa
from prefilter import EnergyPrefilter
from sed.sed_cropper import BarkCropper
from template_manager import TemplateManager
from verifier import DogVerifier


def main():
    # 1. 加载模型
    prefilter = EnergyPrefilter()
    cropper = BarkCropper("models/sed_model")  # 你的 SED 模型
    tm = TemplateManager("models/embed_model")  # 你的 embedding 模型

    # 2. 注册狗（示例）
    tm.register_dog("dog01", ["data/dog01/bark1.wav", "data/dog01/bark2.wav"])
    tm.save_templates("templates.npz")

    # 3. 加载待测音频
    audio, sr = librosa.load("test/unknown_bark.wav", sr=16000)

    # 4. 前端初筛
    events = prefilter.detect_events(audio)
    print(f"初筛检测到 {len(events)} 个高能量事件")

    # 5. 精确裁剪 + 验证
    verifier = DogVerifier("models/embed_model", tm)
    for i, (start, end) in enumerate(events):
        segment = audio[start:end]
        barks = cropper.crop_barks(segment)
        print(f"事件 {i + 1}: 裁剪出 {len(barks)} 段狗吠")

        for j, bark in enumerate(barks):
            is_match, dog_id, sim = verifier.verify(bark, target_dog_id="dog01")
            print(f"  → 吠叫 {j + 1}: {'✅ 是 dog01' if is_match else '❌ 不是'} (相似度: {sim:.4f})")


if __name__ == "__main__":
    main()