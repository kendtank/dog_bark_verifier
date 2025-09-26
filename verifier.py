# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午4:35
@Author  : Kend
@FileName: verifier.py
@Software: PyCharm
@modifier:
验证识别类
"""


import numpy as np
from scipy.spatial.distance import cosine
from utils import extract_logmel


class DogVerifier:
    def __init__(self, embed_model_path, template_manager, sr=16000, threshold=0.75):
        self.embed_model = tf.saved_model.load(embed_model_path)
        self.tm = template_manager
        self.sr = sr
        self.threshold = threshold

    def verify(self, bark_audio, target_dog_id=None):
        """
        验证一段狗吠
        :param bark_audio: numpy array
        :param target_dog_id: 如果指定，只验证是否是该狗；否则返回最相似狗
        :return: (is_match, dog_id, similarity)
        """
        mel = extract_logmel(bark_audio, sr=self.sr)
        mel = np.expand_dims(mel, axis=(0, -1))
        emb = self.embed_model(mel).numpy().flatten()
        emb = emb / np.linalg.norm(emb)

        if target_dog_id:
            if target_dog_id not in self.tm.templates:
                return False, None, 0.0
            sim = 1 - cosine(emb, self.tm.templates[target_dog_id])
            return sim >= self.threshold, target_dog_id, sim
        else:
            best_dog, best_sim = None, -1.0
            for dog_id, template in self.tm.templates.items():
                sim = 1 - cosine(emb, template)
                if sim > best_sim:
                    best_sim = sim
                    best_dog = dog_id
            return best_sim >= self.threshold, best_dog, best_sim