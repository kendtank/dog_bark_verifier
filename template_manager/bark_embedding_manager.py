# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/29 下午3:43
@Author  : Kend
@FileName: bark_embedding_manager.py
@Software: PyCharm
@modifier:
"""
from sed.sed_cropper import add_random_noise_like

"""
狗吠声纹注册与管理模块(面向产品的设计， 狗吠声纹验证非狗吠识别)
目标
    - 用于存储/匹配狗吠 embedding
    - 支持 Flash 持久化
    - MCU 友好（低内存、无动态分配）
当前实现：
    - 使用 .npy 文件存储
    - 适合 Python 算法验证阶段
"""


import os
import numpy as np
import torch
import librosa
from scipy.spatial.distance import cosine
from template_manager.tiny_dog_embeddingNet import TinyDogEmbeddingNet
from prefilter import load_audio


class BarkEmbeddingManager:
    """负责模板管理、embedding提取、保存/加载"""
    def __init__(self,
                 model_path="best_dog_embedding.pth",
                 embedding_dim=16,
                 max_templates_per_dog=10,
                 similarity_threshold=0.7,
                 storage_path="bark_templates.npy",
                 device='cpu'):
        # 预处理参数
        self.sample_rate = 16000
        self.target_duration = 0.4
        self.n_mels = 32
        self.n_fft = 400
        self.hop_length = 200
        self.target_frames = 32

        # 模型参数
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.max_templates_per_dog = max_templates_per_dog
        self.similarity_threshold = similarity_threshold
        self.storage_path = storage_path
        self.device = device

        self.model = None
        self.dogs = {}  # {dog_id: [emb1, emb2, ...]}

        # 初始化
        self._load_model()
        self.load_templates()

    def _load_model(self):
        """加载 PyTorch embedding 模型"""
        self.model = TinyDogEmbeddingNet(embedding_dim=self.embedding_dim)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"加载 embedding 模型 成功: {self.model_path}")


    def _crop_audio(self, y, target_duration=None):
        """裁剪或者填充预处理音频片段"""
        if target_duration is None:
            target_duration = self.target_duration
        target_len = int(self.sample_rate * target_duration)

        # 大于目标时长： 截取中心片段
        if len(y) > target_len:
            # 中心裁剪
            start = (len(y) - target_len) // 2
            y = y[start:start + target_len]
        elif len(y) < target_len:
            # 小于目标时长： 中心对齐 + 补随机噪声
            # 中心对齐 + 补随机噪声
            pad_total = target_len - len(y)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left

            # 生成随机噪声（幅度基于信号RMS）
            rms = np.sqrt(np.mean(y ** 2)) if np.any(y) else 1e-6
            noise_level = rms * 0.1  # 噪声为信号的10%
            left_noise = add_random_noise_like(np.zeros(pad_left), noise_level)
            right_noise = add_random_noise_like(np.zeros(pad_right), noise_level)
            y = np.concatenate([left_noise, y, right_noise])
        return y


    def preprocess_wav(self, wav_path_or_array):
        """ 提取log-mel特征 """
        """ 产品的逻辑是只需要处理内存中加载好的音频 这里我们做demo需要考虑音频路径"""
        y, _ = load_audio(wav_path_or_array, target_sr=self.sample_rate)
        y = self._crop_audio(y, target_duration=self.target_duration)

        # 获取log-mel
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # 归一化
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
        # 固定时间帧
        mel_db = librosa.util.fix_length(mel_db, size=self.target_frames, axis=1)
        return mel_db.astype(np.float32)


    def extract_embedding_from_wav(self, wav_path_or_array):
        """从音频路径或 NumPy 波形中提取归一化 embedding 向量"""
        mel = self.preprocess_wav(wav_path_or_array)
        mel = torch.tensor(mel, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 32)

        with torch.no_grad():
            emb = self.model(mel)
        emb = emb.cpu().numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb


    def register_dog_from_wav_folder(self, dog_id, wav_folder):
        """
        从本地音频文件夹注册狗吠声纹（支持多种格式）
        :param dog_id: 狗ID
        :param wav_folder: 包含多个狗吠 WAV 的文件夹
        """
        print(f"正在注册狗 {dog_id} ...")
        # 支持的常见音频格式
        SUPPORTED_EXTS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.mp4')
        # 遍历文件夹
        all_files = [f for f in os.listdir(wav_folder) if f.lower().endswith(SUPPORTED_EXTS)]

        if not all_files:
            raise ValueError(f"{wav_folder} 中没有可识别的音频文件")

        embeddings = []

        for fname in sorted(all_files):
            wav_path = os.path.join(wav_folder, fname)
            try:
                emb = self.extract_embedding_from_wav(wav_path)
                embeddings.append(emb)
                print(f"完成提取embedding：{fname}")
            except Exception as e:
                print(f"提取失败： {fname}: {type(e).__name__} - {e}")

        if not embeddings:
            raise ValueError(f"{wav_folder} 中未成功提取任何 embedding")

        # 添加或更新模板库
        if dog_id not in self.dogs:
            self.dogs[dog_id] = []

        # FIFO 限制数量
        new_embs = embeddings[:self.max_templates_per_dog]
        total = self.dogs[dog_id] + new_embs
        self.dogs[dog_id] = total[-self.max_templates_per_dog:]

        self.save_templates()
        print(f"注册完成: {dog_id} ({len(new_embs)} 新模板, 总计 {len(self.dogs[dog_id])})")


    def register_dog_from_online_wav(self, dog_id, wav_list: list):
        """
        从在线音频流注册狗吠声纹
        :param dog_id: 狗ID
        :param wav_list: 包含多条音频的列表，可以是:
                         - 文件路径 str
                         - numpy 数组
                         - 二进制 bytes (wav/mp3)
        """
        print(f"正在从在线数据注册狗....: {dog_id}")

        embeddings = []

        for idx, data in enumerate(wav_list):
            try:
                # 判断输入类型
                if isinstance(data, str):
                    emb = self.extract_embedding_from_wav(data)
                elif isinstance(data, np.ndarray):
                    emb = self.extract_embedding_from_wav(data)
                elif isinstance(data, (bytes, bytearray)):
                    # 使用 soundfile 从内存读取
                    import soundfile as sf
                    import io
                    y, sr = sf.read(io.BytesIO(data))
                    if sr != self.sample_rate:
                        y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                    emb = self.extract_embedding_from_wav(y)
                else:
                    raise TypeError(f"不支持的输入类型: {type(data)}")
                print(f"建立模版的embing特征维度：{emb.shape}")
                embeddings.append(emb)
                print(f"在线音频 {idx+1}/{len(wav_list)} 注册成功")
            except Exception as e:
                print(f"在线音频 {idx+1}: {type(e).__name__} - {e}")

        if not embeddings:
            raise ValueError("未成功提取任何在线 embedding")

        # 添加或更新模板
        if dog_id not in self.dogs:
            self.dogs[dog_id] = []

        new_embs = embeddings[:self.max_templates_per_dog]
        total = self.dogs[dog_id] + new_embs
        self.dogs[dog_id] = total[-self.max_templates_per_dog:]

        self.save_templates()
        print(f"✅ 在线注册完成: {dog_id} ({len(new_embs)} 新模板, 总计 {len(self.dogs[dog_id])})")



    def is_known_dog_from_wav(self, wav_path, dog_id=None):
        """
        判断 WAV 是否为已知狗吠
        :param dog_id: 如果指定，只匹配该狗
        :return: (is_known, matched_dog_id, max_similarity)
        """
        if not self.dogs:
            return False, None, 0.0

        query_emb = self.extract_embedding_from_wav(wav_path)
        max_sim = -1.0
        matched_dog = None

        dogs_to_check = [dog_id] if dog_id else self.dogs.keys()

        for did in dogs_to_check:
            if did not in self.dogs:
                continue
            for ref_emb in self.dogs[did]:
                sim = 1 - cosine(query_emb, ref_emb)  # 余弦相似度
                if sim > max_sim:
                    max_sim = sim
                    matched_dog = did

        is_known = max_sim >= self.similarity_threshold
        return is_known, matched_dog, max_sim

    def delete_dog(self, dog_id):
        if dog_id in self.dogs:
            del self.dogs[dog_id]
            self.save_templates()
            print(f"删除声纹: {dog_id}")

    def list_dogs(self):
        return list(self.dogs.keys())

    def save_templates(self):
        """保存为 .npy"""
        serializable = {
            'config': {
                'model_path': self.model_path,
                'embedding_dim': self.embedding_dim,
                'max_templates_per_dog': self.max_templates_per_dog,
                'similarity_threshold': self.similarity_threshold
            },
            'dogs': {
                dog_id: [emb.tolist() for emb in embs]
                for dog_id, embs in self.dogs.items()
            }
        }
        np.save(self.storage_path, serializable)
        print(f"声纹已保存到: {self.storage_path}")

    def load_templates(self):
        """从 .npy 加载"""
        if os.path.exists(self.storage_path):
            try:
                data = np.load(self.storage_path, allow_pickle=True).item()
                config = data.get('config', {})
                self.model_path = config.get('model_path', self.model_path)
                self.embedding_dim = config.get('embedding_dim', self.embedding_dim)
                self.max_templates_per_dog = config.get('max_templates_per_dog', self.max_templates_per_dog)
                self.similarity_threshold = config.get('similarity_threshold', self.similarity_threshold)

                self.dogs = {
                    dog_id: [np.array(emb, dtype=np.float32) for emb in embs]
                    for dog_id, embs in data.get('dogs', {}).items()
                }
                return self.dogs
            except Exception as e:
                print(f"加载失败: {e}")
                return {}
        else:
            print("初始化空声纹库")


if __name__ == '__main__':
    """ 测试注册狗吠的声纹特征 """

    # ================== 配置测试路径 ==================
    MODEL_PATH = r"D:\work\code\dog_bark_verifier\model\best_dog_embedding.pth"  # 替换为你的模型路径
    TEST_DATA_ROOT = r"D:\work\datasets\tinyML\compare_dog"

    # 目录结构:
    # compare_dog/
    #   ├── dog01/       # 已知狗A的注册样本
    #   │   ├── bark1.wav
    #   │   └── bark2.wav
    #   └── dog02/       # 其他狗的样本（用于测试未知）
    #       └── bark1.wav


    DOG01_FOLDER = os.path.join(TEST_DATA_ROOT, "dog01")
    DOG02_FOLDER = os.path.join(TEST_DATA_ROOT, "dog02")

    # 创建测试输出目录
    TEST_OUTPUT = "test_bark_manager"
    os.makedirs(TEST_OUTPUT, exist_ok=True)
    STORAGE_PATH = os.path.join(TEST_OUTPUT, "test_templates.npy")

    # ================== 1. 初始化管理器 ==================
    print("正在初始化 BarkEmbeddingManager...")
    try:
        manager = BarkEmbeddingManager(
            model_path=MODEL_PATH,
            embedding_dim=16,
            max_templates_per_dog=3,
            similarity_threshold=0.6,
            storage_path=STORAGE_PATH,
            device='cpu'
        )
        print("初始化成功!\n")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        exit(1)

    # ================== 2. 测试单文件 embedding 提取 ==================
    print("测试单文件 embedding 提取...")
    test_wav_files = [f for f in os.listdir(DOG01_FOLDER) if f.endswith('.wav')]
    if test_wav_files:
        test_wav = os.path.join(DOG01_FOLDER, test_wav_files[0])
        try:
            emb = manager.extract_embedding_from_wav(test_wav)
            print(f"成功提取 embedding, shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")
        except Exception as e:
            print(f"提取失败: {e}")
            exit(1)
    else:
        print("跳过单文件测试（无 WAV 文件）")
    print()

    # ================== 3. 注册狗01的声纹 ==================
    print("注册 dog01 的声纹...")
    try:
        manager.register_dog_from_wav_folder("dog01", DOG01_FOLDER)
        print(f"注册完成, 当前狗列表: {manager.list_dogs()}\n")
    except Exception as e:
        print(f"注册失败: {e}")
        exit(1)

    # ================== 4. 验证已知狗吠（dog01） ==================
    print("验证已知狗吠 (dog01)...")
    if test_wav_files:
        is_known, matched_id, sim = manager.is_known_dog_from_wav(test_wav, dog_id="dog01")
        print(f"验证结果: is_known={is_known}, dog_id={matched_id}, similarity={sim:.4f}")
        assert is_known == True, "已知狗吠应被识别为已知!"
        assert matched_id == "dog01", "应匹配到 dog01!"
        print("已知狗吠验证通过!\n")
    else:
        print("⚠跳过已知狗吠验证（无 WAV 文件）\n")

    # ================== 5. 验证未知狗吠（dog02） ==================
    print("验证未知狗吠 (dog02)...")
    dog02_wavs = [f for f in os.listdir(DOG02_FOLDER) if f.endswith('.WAV')] if os.path.exists(DOG02_FOLDER) else []
    if dog02_wavs:
        test_wav02 = os.path.join(DOG02_FOLDER, dog02_wavs[0])
        is_known, matched_id, sim = manager.is_known_dog_from_wav(test_wav02, dog_id="dog01")
        print(f"验证结果: is_known={is_known}, dog_id={matched_id}, similarity={sim:.4f}")
        # 注意: 未知狗可能偶然相似，但通常应 < threshold
        if is_known:
            print("警告: 未知狗被误判为已知 (可能需调整 threshold)")
        else:
            print("未知狗吠验证通过 (正确拒绝)!\n")
    else:
        print("跳过未知狗吠验证（无 dog02 目录或 WAV 文件）\n")

    # ================== 6. 测试声纹删除 ==================
    print("测试声纹删除...")
    manager.delete_dog("dog01")
    assert "dog01" not in manager.list_dogs(), "删除后应不在列表中!"
    print("声纹删除测试通过!\n")

    # ================== 7. 重新注册并测试持久化 ==================
    print("测试持久化保存/加载...")
    manager.register_dog_from_wav_folder("dog01", DOG01_FOLDER)
    original_dogs = manager.list_dogs()

    # 重新初始化（模拟重启）
    manager2 = BarkEmbeddingManager(
        model_path=MODEL_PATH,
        storage_path=STORAGE_PATH,
        device='cpu'
    )

    loaded_dogs = manager2.list_dogs()

    assert set(original_dogs) == set(loaded_dogs), "持久化加载应保持一致!"
    print(f"持久化测试通过! 加载狗列表: {loaded_dogs}\n")

    # ================== 8. 清理测试文件 ==================
    print("清理测试文件...")
    if os.path.exists(STORAGE_PATH):
        os.remove(STORAGE_PATH)
    if os.path.exists(TEST_OUTPUT) and not os.listdir(TEST_OUTPUT):
        os.rmdir(TEST_OUTPUT)
    print("测试完成，所有检查通过!")