# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午2:08
@Author  : Kend
@FileName: tiny_dog_embeddingNet.py
@Software: PyCharm
@modifier:
"""




import torch.nn as nn


class TinyDogEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (1, 32, 32)
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (8, 16, 16)

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # (16, 4, 4)
        )
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim)  # L2 归一化前的稳定层
        )

    def forward(self, x):
        x = self.features(x)
        x = self.embedder(x)
        return nn.functional.normalize(x, p=2, dim=1)  # L2 归一化 embedding