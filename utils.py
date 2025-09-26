# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午4:35
@Author  : Kend
@FileName: utils.py
@Software: PyCharm
@modifier:
"""


import numpy as np
import librosa

# 音频重采样为16000Hz并做单通道
def resample(y, sr):
    return librosa.resample(y, sr, 16000)



def extract_mfcc(y, sr=16000, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    return mfcc.astype(np.float32)


def extract_logmel(y, sr=16000, n_mels=32, target_time=32):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
    if mel_db.shape[1] > target_time:
        mel_db = mel_db[:, :target_time]
    else:
        mel_db = librosa.util.fix_length(mel_db, size=target_time, axis=1)
    return mel_db.astype(np.float32)