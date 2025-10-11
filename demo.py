# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/29 下午4:36
@Author  : Kend
@FileName: demo.py
@Software: PyCharm
@modifier:
主演示脚本
"""


"""
演示步骤：
    1: 输入一段音频模拟用户录制注册声纹模版。算法自动裁剪和生成声纹模版
    2: 输入一段正向相关(有同一只狗的音频)， 算法自动裁剪和对比相似度
    3: 输入一段负相关(有狗吠场景, 但是不是模版注册的狗的音频)， 算法自动裁剪和对比相似度
    4: 输入一段背景音频(包涵相对安静和有类似与狗吠的尖锐声音的音频)， 算法自动裁剪和对比相似度
重点关注结果对象:
    1: 裁剪段召回率
    2: 声纹比对正阳性和假阳性的指标
    3: 背景音频的干扰, 测试算法的鲁棒性
"""

import psutil
import time
import os

# 获取初始状态
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024
print(f"初始内存使用: {initial_memory:.2f} MB")
start_time = time.time()




from prefilter import preprocess_for_bark_detection
from sed import BarkSEDRefiner
from template_manager import BarkEmbeddingManager
from verifier import DogBarkVerifier
from utils import save_audio

# 测试参数
wav_path = "data/ori_denoise_audio/outdoor_braking_01.mp3"
# wav_path = r"D:\work\code\dog_bark_verifier\data\dog_braking_test.WAV"
dog_bark_model = "model/tiny_bark_cnn_v3.tflite"
emd_model = "model/best_dog_embedding.pth"

# 阈值参数
prefilter_confidence_threshold = 0.5  # 音频初筛阈值
verifier_threshold = 0.6  # 声纹比对阈值


def main(function="manage", dog_id="dog01"):
    """
    :param function: 功能选择， 有实时推理检测和模版管理两类
    :param dog_id: 狗狗id， 实际产品只支持一只狗
    :return:
    """
    """
    四个功能模块的初始化， 前端初筛模块， SED细筛模块， 特征模版管理模块， 实时流验证器模块
    """
    sed_refiner = BarkSEDRefiner(tflite_model_path=dog_bark_model,
                                 confidence_threshold=prefilter_confidence_threshold)  # SED模型
    tm = BarkEmbeddingManager(emd_model, similarity_threshold=verifier_threshold)  # embedding模版管理
    verifier = DogBarkVerifier(emd_model, threshold=verifier_threshold)
    print(f"模块初始化后内存: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 预处理音频， 获取疑似狗吠的片段
    y_denoised, candidates, _ = preprocess_for_bark_detection(wav_path)
    print(f"前端初筛模块： 检测到 {len(candidates)} 个疑似狗吠片段...移交到细筛处理模块")
    print(f"音频预处理后内存: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 保存起来初筛的狗吠片段
    for i, (start, end) in enumerate(candidates):
        save_audio(y_denoised[start:end], 16000, f"template_manager/preprocess_for_bark_detection/{dog_id}_{i}.wav")

    # debug--
    # print(candidates)  # list
    # print(y_denoised)  # numpy.ndarray
    # 2. SED 精修
    precise_barks = sed_refiner.refine_all_candidates(y_denoised, candidates)
    if len(precise_barks) <= 1:
        print("细筛模块过滤中, 音频无有效狗吠...")
        return
    print(f"细筛模块过滤中, 音频有 {len(precise_barks)} 个有效狗吠片段")
    print(f"SED精修后内存: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    if function == "detect":
        # 检查模版库是否已经注册
        if not tm.list_dogs() or 0 == 1:
            print("模版库为空，请先注册模版")  # 实际产品测试中，需要先完成模版管理模块的功能测试， 这里脚本测试内存中永远是为空的
            # return
        print("开始对输入音频做模版库声纹验证...")
        for i, (start, end) in enumerate(precise_barks):
            segment = y_denoised[start:end]
            # 验证该段狗吠是不是属于该只主人狗  # TODO: 需要先测试和完成验证器模块的功能测试-1010
            dog_id, sim = verifier.verify_audio(segment)
            if dog_id is None:
                print(f"  → 吠叫 {i + 1}无匹配狗, 最大匹配度{sim:.4f}")
                continue
            print(f"  → 吠叫 {i + 1}属于 {dog_id} 狗, 匹配度: {sim:.4f}")

    elif function == "manage":
        # 管理模版 注册狗
        dog_wav_list = []
        print("开始对输入音频做模版库声纹特征建立...")
        for i, (start, end) in enumerate(precise_barks):
            segment = y_denoised[start:end]
            # 保存狗吠片段
            save_audio(segment, 16000, f"template_manager/dog_wav/{dog_id}_{i}.wav")
            dog_wav_list.append(segment)
        tm.register_dog_from_online_wav(dog_id, dog_wav_list)

    else:
        print("请选择功能：detect/manage, 不支持其他功能输入")

    # 在程序结束前添加总结
    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"最终内存使用: {final_memory:.2f} MB")
    print(f"内存增长: {final_memory - initial_memory:.2f} MB")
    print(f"总执行时间: {end_time - start_time:.4f} 秒")


if __name__ == "__main__":
    main(
        function="manage",
        dog_id="dog01"
    )
