# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午4:33
@Author  : Kend
@FileName: template_manager.py
@Software: PyCharm
@modifier:
模板注册类
"""


import os
import numpy as np
import tensorflow as tf
from .utils import extract_logmel


class TemplateManager:
    def __init__(self, embed_model_path, sr=16000):
        self.embed_model = tf.saved_model.load(embed_model_path)
        self.sr = sr
        self.templates = {}  # {dog_id: embedding}

    def register_dog(self, dog_id, audio_files):
        """注册一只狗：输入多段原始吠叫，生成模板"""
        embeddings = []
        for file in audio_files:
            y, _ = librosa.load(file, sr=self.sr, mono=True)
            mel = extract_logmel(y, sr=self.sr)
            mel = np.expand_dims(mel, axis=(0, -1))  # (1, 32, 32, 1)
            emb = self.embed_model(mel).numpy().flatten()
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        # 平均模板
        avg_emb = np.mean(embeddings, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        self.templates[dog_id] = avg_emb.astype(np.float32)
        print(f"✅ 注册 {dog_id}: 基于 {len(embeddings)} 段吠叫")

    def save_templates(self, path):
        np.savez_compressed(path, **self.templates)

    def load_templates(self, path):
        data = np.load(path)
        self.templates = {k: v for k, v in data.items()}