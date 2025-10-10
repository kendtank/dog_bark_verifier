# -*- coding: utf-8 -*-
"""
@Time    : 2025/10/9 下午4:08
@Author  : Kend
@FileName: dog_bark_verifier.py
@Software: PyCharm
@modifier:
"""


"""
声纹验证具体实现类
适用于实时/离线音频推理场景
"""
import os
import numpy as np
import soundfile as sf
from template_manager import BarkEmbeddingManager


class DogBarkVerifier(BarkEmbeddingManager):
    """声纹验证具体实现类"""
    def __init__(self,
                 model_path,
                 templates_path="bark_templates.npy",
                 threshold=0.75,
                 smooth_window=3,
                 device='cpu'):
        super().__init__(model_path=model_path,
                         storage_path=templates_path,
                         device=device)
        self.threshold = threshold              # 声纹相似度阈值
        self.smooth_window = smooth_window      # 实时验证时的平滑窗口大小
        self.recent_scores = []                 # 缓存连续的置信度分数


    # 基础验证（单段音频）
    def verify_audio(self, wav_input):
        """
        验证一段音频是否属于已注册的狗
        :param wav_input: 音频路径或 np.ndarray
        :return: (matched_dog_id, similarity)
        """
        # 提取声纹 embedding
        # emb = self.preprocess_wav(wav_input)
        # 提取声纹 embedding (应该使用 extract_embedding_from_wav 而不是 preprocess_wav)
        emb = self.extract_embedding_from_wav(wav_input)

        if emb is None:
            print("提取声纹 embedding 失败")
            return None, 0.0
        # 计算相似度
        # print(f"实时提取特征维度: {emb.shape}")  # 调试信息
        best_dog, best_score = self._match_embedding(emb)
        # 判断阈值
        if best_score >= self.threshold:
            return best_dog, float(best_score)
        else:
            return None, float(best_score)

    # =========================================================
    # 🐾 2. 实时流式验证（连续帧）
    # =========================================================
    def verify_stream(self, wav_chunk, dog_id=None):
        """
        流式验证模式：输入短音频块（例如 0.3~0.5 秒）
        可用于实时麦克风推理
        :param wav_chunk: np.ndarray 格式的单通道音频数据
        :param dog_id: 需要验证声纹的狗狗的id， 若为None直接对比整个模版库, 若指定了直接对比该dog_id的特征库
        :return: (matched_dog_id, smoothed_similarity)
        """
        emb = self.extract_embedding_from_wav(wav_chunk)
        if emb is None:
            return None, 0.0
        if dog_id is not None:
            return self._match_embedding_by_dog_id(emb, dog_id)
        else:

            dog_id, score = self._match_embedding(emb)

        # 更新平滑窗口
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.smooth_window:
            self.recent_scores.pop(0)
        avg_score = np.mean(self.recent_scores)

        if avg_score >= self.threshold:
            return dog_id, float(avg_score)
        else:
            return None, float(avg_score)


    def _match_embedding(self, emb):
        """计算当前 embedding 与模板库的相似度  这里采用平均相似度 全量匹配 声纹特征库匹配 """
        # print(f"查询特征维度: {emb.shape}")
        best_score = 0
        best_dog = None

        for dog_id, templates in self.dogs.items():
            sims = [self.cosine_similarity(emb, t) for t in templates]
            avg_sim = np.mean(sims)
            print(f"{dog_id} 狗的声纹相似度：{avg_sim}")
            if avg_sim > best_score:
                best_score = avg_sim
                best_dog = dog_id
        # 取最高相似度的狗
        return best_dog, best_score

    def _match_embedding_by_dog_id(self, emb, dog_id):
        """计算当前 embedding 与指定狗的模板库的相似度"""
        # 判断是否注册该狗狗
        if dog_id not in self.dogs:
            print(f"未注册的狗狗：{dog_id}")
            return 0.0
        if len(self.dogs[dog_id]) == 0:
            print(f"{dog_id} 狗的声纹模板为空")
            return 0.0
        templates = self.dogs[dog_id]
        sims = [self.cosine_similarity(emb, t) for t in templates]
        print(f"{dog_id} 狗的声纹相似度：{sims}")
        avg_sim = np.mean(sims)
        return avg_sim


    @staticmethod
    def cosine_similarity(a, b):
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
