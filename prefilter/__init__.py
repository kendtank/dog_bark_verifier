# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午5:31
@Author  : Kend
@FileName: __init__.py.py
@Software: PyCharm
@modifier:
"""

"""
前端预处理音频模块
    主要实现功能：
        - 对原始音频做重采样和单声道的处理
        - 对重采样的音频做降噪和增强处理
        - 对预处理好的音频做前端初筛（能量/突发检测）
        - 完成模块功能测试以及预处理的音频的特征损失
"""


from .bark_frontend import preprocess_for_bark_detection



