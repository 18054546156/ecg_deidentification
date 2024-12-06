import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal as sig
import pywt
import logging
from tqdm import tqdm
import os
import wfdb
import pandas as pd
import torch.nn.functional as F
from scipy.signal import hilbert, find_peaks
from scipy.stats import pearsonr
import math
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_mitdb():
    """下载MIT-BIH数据集"""
    data_path = '../data/raw/mit/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        logger.info("Downloading MIT-BIH dataset...")
        wfdb.dl_database('mitdb', data_path)
        logger.info("Download completed!")
    else:
        logger.info("MIT-BIH dataset already exists.")
    return data_path


def load_record(record_path):
    """加载单条记录"""
    try:
        # 读取信号数据
        record = wfdb.rdrecord(record_path)
        # 读取标注数据
        annotation = wfdb.rdann(record_path, 'atr')

        # 获取ECG信号（通常使用第一导联）
        ecg_signal = record.p_signal[:, 0]

        # 获取标注信息
        r_peaks = annotation.sample  # R波位置
        beat_types = annotation.symbol  # 心跳类型

        return {
            'signal': ecg_signal,
            'r_peaks': r_peaks,
            'beat_types': beat_types,
            'patient_id': record_path.split('/')[-1]  # 从路径提取记录ID
        }
    except Exception as e:
        logger.error(f"Error loading record {record_path}: {str(e)}")
        return None


def prepare_dataset():
    """准备数据集"""
    # 1. 下载数据
    data_path = download_mitdb()

    # 2. 获取所有记录
    record_paths = []
    for file in os.listdir(data_path):
        if file.endswith('.dat'):
            record_path = os.path.join(data_path, file[:-4])
            record_paths.append(record_path)

    # 3. 加载所有记录
    dataset = []
    for record_path in tqdm(record_paths, desc="Loading records"):
        record_data = load_record(record_path)
        if record_data is not None:
            dataset.append(record_data)

    logger.info(f"Loaded {len(dataset)} records successfully.")
    return dataset


class ECGProcessor:
    def __init__(self, window_sec=10, overlap_sec=5, sampling_rate=360):
        """
        初始化ECG处理器
        Args:
            window_sec: 窗口长度(秒)
            overlap_sec: 重叠长度(秒)
            sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_sec * sampling_rate
        self.overlap = overlap_sec * sampling_rate

        # 心跳波形参数 (基于生理学特征)
        self.p_qrs_t_duration = int(0.6 * sampling_rate)  # 典型P-QRS-T持续时间约600ms
        self.beat_types = {
            'N': 0,  # 正常心跳
            'S': 1,  # 室上性早搏
            'V': 2,  # 室性早搏
            'F': 3,  # 融合心跳
            'Q': 4  # 未知类型
        }

    def extract_windows(self, ecg_signal, annotations):
        """
        提取ECG窗口并进行标注，确保心跳的完整性
        Args:
            ecg_signal: 原始ECG信号
            annotations: 包含r_peaks, beat_types和patient_id的字典
        Returns:
            windows: 提取的窗口列表
            heart_labels: 心跳类型标签列表（多标签）
            id_labels: 病人ID标签列表
        """
        windows = []
        heart_labels = []
        id_labels = []

        # 计算有效的窗口起始位置
        stride = self.window_size - self.overlap
        signal_length = len(ecg_signal)

        # 确保最后一个窗口不会超出信号范围
        start_points = range(0, signal_length - self.window_size + 1, stride)

        for start in start_points:
            end = start + self.window_size

            # 获取当前窗口的标注信息
            window_annotations = self._get_window_annotations(
                annotations, start, end)

            # 检查窗口完整性
            if self._check_window_completeness(window_annotations):
                # 提取窗口
                window = ecg_signal[start:end]

                # 生成标签
                heart_label = self._get_heart_labels(window_annotations)
                id_label = annotations['patient_id']

                windows.append(window)
                heart_labels.append(heart_label)
                id_labels.append(id_label)

        return windows, heart_labels, id_labels

    def _get_window_annotations(self, annotations, start, end):
        """
        获取窗口内的标注信息，包括完整的心跳
        """
        window_annotations = {
            'r_peaks': [],
            'beat_types': [],
            'patient_id': annotations['patient_id']
        }

        half_beat = self.p_qrs_t_duration // 2
        extended_start = start + half_beat  # 确保第一个心跳完整
        extended_end = end - half_beat  # 确保最后一个心跳完整

        # 获取扩展窗口范围内的所有R波
        for i, peak in enumerate(annotations['r_peaks']):
            # 只包含完整的心跳
            if extended_start <= peak < extended_end:
                # 检查前后是否有足够空间容纳完整心跳
                prev_peak = annotations['r_peaks'][i - 1] if i > 0 else peak - self.p_qrs_t_duration
                next_peak = annotations['r_peaks'][i + 1] if i < len(
                    annotations['r_peaks']) - 1 else peak + self.p_qrs_t_duration

                # 确保与相邻心跳不重叠
                if (peak - prev_peak >= self.p_qrs_t_duration and
                        next_peak - peak >= self.p_qrs_t_duration):
                    window_annotations['r_peaks'].append(peak - start)
                    window_annotations['beat_types'].append(annotations['beat_types'][i])

        return window_annotations

    def _check_window_completeness(self, annotations):
        """
        检查窗口内心跳的完整性
        返回:
            bool: 窗口是否包含完整的心跳
        """
        r_peaks = annotations['r_peaks']

        if len(r_peaks) < 1:  # 至少需要一个完整心跳
            return False

        # 检查第一个和最后一个心跳是否完整
        if (r_peaks[0] < self.p_qrs_t_duration // 2 or
                self.window_size - r_peaks[-1] < self.p_qrs_t_duration // 2):
            return False

        # 检查相邻心跳之间的间隔
        for i in range(1, len(r_peaks)):
            if r_peaks[i] - r_peaks[i - 1] < self.p_qrs_t_duration:
                return False

        return True

    def _get_heart_labels(self, annotations):
        """
        生成心跳类型的多标签编码
        """
        label_vector = np.zeros(len(self.beat_types))
        beat_types = annotations['beat_types']

        # 统计每种类型的心跳数量
        for beat_type in beat_types:
            if beat_type in self.beat_types:
                label_vector[self.beat_types[beat_type]] = 1

        return label_vector

    def preprocess_signal(self, signal):
        """
        预处理ECG信号
        1. 基线漂移校正
        2. 带通滤波 (0.5-40Hz)
        3. 标准化
        """
        # 1. 基线漂移校正
        coeffs = pywt.wavedec(signal, 'db4', level=9)
        coeffs[0] = np.zeros_like(coeffs[0])
        signal = pywt.waverec(coeffs, 'db4')

        # 2. 带通滤波
        nyquist_freq = self.sampling_rate / 2
        low = 0.5 / nyquist_freq
        high = 40.0 / nyquist_freq
        b, a = sig.butter(4, [low, high], btype='band')
        signal = sig.filtfilt(b, a, signal)

        # 3. 标准化
        signal = (signal - np.mean(signal)) / np.std(signal)

        return signal


class ECGDataset(Dataset):
    def __init__(self, data_path=None, window_sec=10, overlap_sec=5, sampling_rate=360):
        """
        初始化数据集
        Args:
            data_path: 数据路径，如果为None则自动下载
            window_sec: 窗口长度(秒)
            overlap_sec: 重叠长度(秒)
            sampling_rate: 采样率(Hz)
        """
        self.processor = ECGProcessor(window_sec, overlap_sec, sampling_rate)

        # 下载或加载数据
        if data_path is None:
            self.dataset = prepare_dataset()
        else:
            self.dataset = self._load_existing_data(data_path)

        # 处理数据集
        self.windows = []
        self.labels_heart = []
        self.labels_id = []
        self._process_dataset()

        # 转换为numpy数组
        self.windows = np.array(self.windows)
        self.labels_heart = np.array(self.labels_heart)
        self.labels_id = np.array(self.labels_id)

        # 创建ID到索引的映射
        unique_ids = np.unique(self.labels_id)
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
        self.labels_id = np.array([self.id_to_idx[id_] for id_ in self.labels_id])

    def _process_dataset(self):
        """处理数据集中的所有记录"""
        for record in tqdm(self.dataset, desc="Processing records"):
            # 提取窗口
            windows, labels_heart, labels_id = self.processor.extract_windows(
                record['signal'],
                {
                    'r_peaks': record['r_peaks'],
                    'beat_types': record['beat_types'],
                    'patient_id': record['patient_id']
                }
            )

            # 预处理每个窗口
            windows = [self.processor.preprocess_signal(w) for w in windows]

            self.windows.extend(windows)
            self.labels_heart.extend(labels_heart)
            self.labels_id.extend(labels_id)

        # 转换为numpy数组
        self.windows = np.array(self.windows)
        self.labels_heart = np.array(self.labels_heart)
        self.labels_id = np.array(self.labels_id)

    def __getitem__(self, idx):
        """返回元组格式的数据，与work-backup2.py保持一致"""
        return (
            torch.FloatTensor(self.windows[idx]).unsqueeze(0),  # [1, signal_length]
            torch.FloatTensor(self.labels_heart[idx]),
            torch.LongTensor([self.labels_id[idx]])
        )

    def __len__(self):
        return len(self.windows)

    def _load_existing_data(self, data_path):
        """加载已存在的数据"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist")

        dataset = []
        for file in os.listdir(data_path):
            if file.endswith('.dat'):
                record_path = os.path.join(data_path, file[:-4])
                record_data = load_record(record_path)
                if record_data is not None:
                    dataset.append(record_data)

        return dataset


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


class ECGClassifier(nn.Module):
    def __init__(self, num_classes, is_multilabel=True):
        super(ECGClassifier, self).__init__()

        # 特征提取
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(1, 64, kernel_size=50, stride=1, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            AttentionBlock(64),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # 第二层卷积
            nn.Conv1d(64, 128, kernel_size=25, stride=1, padding=12),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            AttentionBlock(128),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # 第三层卷积
            nn.Conv1d(128, 256, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            AttentionBlock(256),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 * 450, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.is_multilabel = is_multilabel

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.is_multilabel:
            x = torch.sigmoid(x)
        return x


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def compute_frequency_importance(model, loader, fs=360, freq_bins=10, device='cpu'):
    model.eval()
    total_importance = None

    # 定义频段边界
    freq_edges = np.linspace(0, fs // 2, freq_bins + 1)

    for data, labels, _ in loader:
        data = data.to(device)
        labels = labels.to(device)

        # 如果是 one-hot 编码，转换为类别索引
        if labels.dim() > 1 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        data.requires_grad = True

        # FFT 分解
        freq_axis = torch.fft.rfftfreq(data.size(-1), d=1 / fs).to(device)

        # 预测并计算梯度
        outputs = model(data)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        # 计算频率梯度
        grad = data.grad
        grad_freq = torch.fft.rfft(grad, dim=-1).abs().mean(dim=0)

        if total_importance is None:
            total_importance = torch.zeros(freq_bins, device=device)

        for i in range(freq_bins):
            band_mask = (freq_axis >= freq_edges[i]) & (freq_axis < freq_edges[i + 1])
            total_importance[i] += grad_freq[:, band_mask].sum().item()

    total_importance /= total_importance.sum()
    return total_importance.cpu().numpy(), freq_edges


def compute_freq_guidance_mask(freq_importance_id, freq_importance_heart, device):
    """
    根据频率重要性计算指导掩模。
    """
    if len(freq_importance_id) != len(freq_importance_heart):
        raise ValueError("Frequency importance lengths must match.")

    freq_guidance = freq_importance_id - freq_importance_heart
    freq_guidance[freq_guidance < 0] = 0

    return torch.tensor(freq_guidance, dtype=torch.float32).to(device)


def plot_frequency_heatmap(freq_importance_id, freq_importance_heart, freq_edges, save_path="frequency_heatmap.png"):
    """
    绘制频率域特征重要性的热力图。
    """
    if len(freq_edges) == len(freq_importance_id) + 1:
        freq_labels = [f"{freq_edges[i]:.1f}-{freq_edges[i + 1]:.1f} Hz" for i in range(len(freq_edges) - 1)]
    else:
        raise ValueError("Frequency edges do not match importance length.")

    plt.figure(figsize=(20, 6))
    x = np.arange(len(freq_labels))

    plt.bar(x - 0.2, freq_importance_id, width=0.4, label="ID Model Importance", color="blue")
    plt.bar(x + 0.2, freq_importance_heart, width=0.4, label="Heart Model Importance", color="orange")

    plt.xticks(x, freq_labels, rotation=45, ha='right')
    plt.title("Frequency Importance for ID and Heart Models")
    plt.ylabel("Normalized Importance")
    plt.xlabel("Frequency Range (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Frequency heatmap saved to {save_path}")


class ECGPrivacyProcessor(nn.Module):
    def __init__(self, input_dim, freq_guidance, noise_limit, device):
        """
        隐私处理器，支持动态频段掩模生成噪声。
        Args:
            input_dim: 输入信号长度
            freq_guidance: 频段指导掩模
            noise_limit: 噪声幅度限制
            device: 设备
        """
        super(ECGPrivacyProcessor, self).__init__()
        self.input_dim = input_dim
        self.freq_guidance = freq_guidance.to(device)
        self.noise_limit = noise_limit

        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)

        self.encoder3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # 解码器
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        # 编码器部分
        enc1 = self.encoder1(x)
        enc2 = self.pool1(enc1)
        enc2 = self.encoder2(enc2)
        enc3 = self.pool2(enc2)
        enc3 = self.encoder3(enc3)

        # 解码器部分
        dec3 = self.decoder3(enc3)
        dec2 = self.decoder2(dec3 + enc2)
        decoded = self.decoder1(dec2 + enc1)

        # 动态频率掩模
        noise_freq = torch.fft.rfft(decoded, dim=-1)
        freq_guidance_resized = F.interpolate(
            self.freq_guidance.unsqueeze(0).unsqueeze(0),
            size=noise_freq.size(-1),
            mode='linear',
            align_corners=True
        ).squeeze(0).squeeze(0)
        masked_freq = noise_freq * freq_guidance_resized
        noise = torch.fft.irfft(masked_freq, n=self.input_dim, dim=-1)

        # 噪声幅度限制
        noise_norm = torch.norm(noise, p=2, dim=-1, keepdim=True)
        scaling_factor = torch.clamp(noise_norm / self.noise_limit, min=1.0)
        noise = noise / scaling_factor

        return noise


class DiffusionPrivacyProcessor(nn.Module):
    """基于扩散模型的ECG隐私处理器"""

    def __init__(self, signal_length=3600, time_steps=1000):
        super().__init__()

        self.signal_length = signal_length
        self.time_steps = time_steps

        # 噪声预测网络
        self.noise_predictor = nn.Sequential(
            # 编码器
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            # 注意力层
            AttentionBlock(128),

            # 解码器
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
        )

        # 生成beta schedule
        self.beta = self._linear_beta_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def _linear_beta_schedule(self):
        """生成线性beta schedule"""
        scale = 1000 / self.time_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.time_steps)

    def _extract(self, a, t, x_shape):
        """从a中提取适当的系数用于时间步t"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, 1, 1).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_hat = torch.sqrt(self._extract(self.alpha_hat, t, x_start.shape))
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self._extract(self.alpha_hat, t, x_start.shape))

        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise

    def p_losses(self, x_start, t, noise=None):
        """计算损失"""
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.noise_predictor(x_noisy)

        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t):
        """单步去噪采样"""
        betas_t = self._extract(self.beta, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            torch.sqrt(1. - self.alpha_hat), t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(
            torch.sqrt(1. / self.alpha), t, x.shape
        )

        # 预测均值
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.noise_predictor(x) / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] > 0:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(betas_t) * noise
        else:
            return model_mean

    @torch.no_grad()
    def forward(self, x):
        """前向传播 - 生成隐私保护后的信号"""
        device = x.device
        batch_size = x.shape[0]

        # 添加少量噪声进行初始化
        x = x + 0.001 * torch.randn_like(x)

        # 逐步去噪
        for t in reversed(range(0, self.time_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)

            # 保持信号的主要特征
            if t < self.time_steps // 2:
                # 使用小波变换保持主要特征
                x_freq = torch.fft.rfft(x.squeeze(1))
                mask = torch.ones_like(x_freq)
                mask[:, int(mask.shape[1] * 0.1):] *= 0.1  # 保持低频成分
                x = torch.fft.irfft(x_freq * mask, n=self.signal_length).unsqueeze(1)

        return x


class ECGPrivacyDataset(Dataset):
    """带有隐私处理的ECG数据集"""

    def __init__(self, original_dataset, privacy_processor=None):
        self.original_dataset = original_dataset
        self.privacy_processor = privacy_processor
        self.preprocessor = ECGProcessor()

    def __getitem__(self, idx):
        # 取原始数据
        x_raw, heart_label, id_label = self.original_dataset[idx]

        # 生成噪声并添加到原始信号
        if self.privacy_processor is not None:
            with torch.no_grad():
                noise = self.privacy_processor(x_raw.unsqueeze(0)).squeeze(0)
                x_processed = x_raw + noise
        else:
            x_processed = x_raw

        # 预处理
        x_preprocessed = self.preprocessor.preprocess_signal(
            x_processed.numpy() if isinstance(x_processed, torch.Tensor)
            else x_processed
        )
        x_preprocessed = torch.FloatTensor(x_preprocessed)

        return {
            'raw': x_raw,
            'processed': x_processed,
            'preprocessed': x_preprocessed,
            'heart_label': heart_label,
            'id_label': id_label
        }

    def __len__(self):
        return len(self.original_dataset)


class PreprocessorFunction(torch.autograd.Function):
    """自定义预处理函数，保持梯度传播"""

    @staticmethod
    def forward(ctx, x, preprocessor):
        # 保存输入用于反向传播
        ctx.save_for_backward(x)

        # 执行预处理
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
            x_preprocessed = preprocessor.preprocess_signal(x_np)
            result = torch.from_numpy(x_preprocessed).float()

        return result.to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的输入
        x, = ctx.saved_tensors

        # 简单传递梯度
        grad_input = grad_output.clone()

        return grad_input, None


class PrivacyProtectionTrainer:
    def __init__(self, processor, heart_model, id_model, device):
        self.processor = processor.to(device)
        self.heart_model = heart_model.to(device)
        self.id_model = id_model.to(device)
        self.device = device

        # 优化器
        self.optimizer = optim.Adam(self.processor.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        # 损失权重
        self.lambda_heart = 1.0  # 心跳分类权重
        self.lambda_id = 10  # ID分类权重
        self.lambda_sim = 1  # 相似度权重
        # self.lambda_noise = 1

        # 初始化最佳指标
        self.best_metrics = None
        self.best_epoch = 0

        # 添加新的损失权重
        self.lambda_diff = 1.0  # 扩散损失的权重

    def compute_similarity(self, x_raw, x_processed):
        """计算波形相似度"""
        # 确保数据在同一设备上并且维度正确
        x_raw = x_raw.to(self.device)
        x_processed = x_processed.to(self.device)

        if len(x_raw.shape) == 2:
            x_raw = x_raw.unsqueeze(1)
        if len(x_processed.shape) == 2:
            x_processed = x_processed.unsqueeze(1)

        # 归一化
        def normalize(x):
            x = x - x.mean(dim=2, keepdim=True)
            return x / (x.std(dim=2, keepdim=True) + 1e-8)

        x_raw_norm = normalize(x_raw)
        x_processed_norm = normalize(x_processed)

        # 计算余弦相似度
        similarity = F.cosine_similarity(
            x_raw_norm.view(x_raw.size(0), -1),
            x_processed_norm.view(x_processed.size(0), -1),
            dim=1
        )

        return similarity.mean()

    def train(self, train_loader, test_loader, epochs=5, patience=30):
        best_overall_score = float('-inf')
        patience_counter = 0

        """修改训练步骤以包含扩散损失"""
        self.processor.train()
        self.heart_model.eval()
        self.id_model.eval()

        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Lambda values - Heart: {self.lambda_heart}, ID: {self.lambda_id}, Sim: {self.lambda_sim}")

        for epoch in range(epochs):
            self.processor.train()
            epoch_metrics = []

            # 训练循环
            for batch in train_loader:
                x_raw, heart_labels, id_labels = [item.to(self.device) for item in batch]

                # # 隐私处理
                # x_processed = self.processor(x_raw)
                # 生成噪声并添加到信号
                # noise = self.processor(x_raw)
                # x_processed = x_raw + noise

                # 随机选择时间步
                t = torch.randint(0, self.processor.time_steps, (x_raw.shape[0],), device=self.device)

                # 计算扩散损失
                diff_loss = self.processor.p_losses(x_raw, t)

                # 生成处理后的信号
                x_processed = self.processor(x_raw)

                # 计算相似度
                similarity = self.compute_similarity(x_raw, x_processed)

                # 模型预测
                heart_pred = self.heart_model(x_processed)
                id_pred = self.id_model(x_processed)

                # 计算损失
                heart_loss = F.binary_cross_entropy_with_logits(heart_pred, heart_labels)
                id_loss = F.cross_entropy(id_pred, id_labels.squeeze())
                sim_loss = 1 - similarity

                # # 噪声正则化
                # noise_loss = torch.mean(noise ** 2)

                # 总损失
                total_loss = (
                        self.lambda_heart * heart_loss -
                        self.lambda_id * id_loss +
                        self.lambda_sim * sim_loss +
                        # self.lambda_noise * torch.mean(noise ** 2)  # 最小化噪声强度
                        self.lambda_diff * diff_loss
                )

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.processor.parameters(), max_norm=1.0)

                self.optimizer.step()

                # 记录指标
                with torch.no_grad():
                    heart_acc = (heart_pred > 0.5).float().eq(heart_labels).float().mean()
                    id_acc = (id_pred.argmax(1) == id_labels.squeeze()).float().mean()

                    epoch_metrics.append({
                        'heart_loss': heart_loss.item(),
                        'id_loss': id_loss.item(),
                        'similarity': similarity.item(),
                        'heart_acc': heart_acc.item(),
                        'id_acc': id_acc.item()
                    })

            # 计算训练epoch平均指标
            train_metrics = {
                k: np.mean([m[k] for m in epoch_metrics])
                for k in epoch_metrics[0].keys()
            }

            # 验证
            val_metrics = self._validate(test_loader)

            # 调整学习率
            self.scheduler.step(val_metrics['heart_acc'])

            # 计算总体得分
            overall_score = (
                    val_metrics['heart_acc'] -
                    val_metrics['id_acc'] +
                    val_metrics['similarity']
            )

            # 保存最佳模型
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                self.best_metrics = val_metrics.copy()
                self.best_epoch = epoch
                patience_counter = 0

                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.processor.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics,
                }, 'best_privacy_processor.pth')

                logger.info("Saved new best model!")
            else:
                patience_counter += 1

            # 输出训练信息
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train - Heart Acc: {train_metrics['heart_acc'] * 100:.1f}%, "
                f"ID Acc: {train_metrics['id_acc'] * 100:.1f}%, "
                f"Sim: {train_metrics['similarity']:.3f} | "
                f"Val - Heart Acc: {val_metrics['heart_acc'] * 100:.1f}%, "
                f"ID Acc: {val_metrics['id_acc'] * 100:.1f}%, "
                f"Sim: {val_metrics['similarity']:.3f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # 提前停止
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logger.info(f"Training completed. Best epoch: {self.best_epoch + 1}")
        return self.best_metrics

    def _validate(self, test_loader):
        """验证模型性能"""
        self.processor.eval()
        metrics_list = []

        with torch.no_grad():
            for batch in test_loader:
                x_raw, heart_labels, id_labels = [item.to(self.device) for item in batch]

                # 隐私处理
                # x_processed = self.processor(x_raw)
                # 生成噪声并添加到信号
                noise = self.processor(x_raw)
                x_processed = x_raw + noise

                # 计算相似度
                similarity = self.compute_similarity(x_raw, x_processed)

                # 模型预测
                heart_pred = self.heart_model(x_processed)
                id_pred = self.id_model(x_processed)

                # 计算损失和准确率
                heart_loss = F.binary_cross_entropy_with_logits(heart_pred, heart_labels)
                id_loss = F.cross_entropy(id_pred, id_labels.squeeze())
                heart_acc = (heart_pred > 0.5).float().eq(heart_labels).float().mean()
                id_acc = (id_pred.argmax(1) == id_labels.squeeze()).float().mean()

                metrics_list.append({
                    'total_loss': (self.lambda_heart * heart_loss -
                                   self.lambda_id * id_loss +
                                   self.lambda_sim * (1 - similarity)).item(),
                    'heart_loss': heart_loss.item(),
                    'id_loss': id_loss.item(),
                    'heart_acc': heart_acc.item(),
                    'id_acc': id_acc.item(),
                    'similarity': similarity.item()
                })

        return {
            k: np.mean([m[k] for m in metrics_list])
            for k in metrics_list[0].keys()
        }


def compute_accuracy(output, target, is_multilabel):
    """计算准确率"""
    if is_multilabel:
        pred = (output > 0.5).float()
        correct = (pred == target).float().mean()
    else:
        pred = output.argmax(dim=1)
        correct = (pred == target).float().mean()
    return correct.item()


def train_classifier(model, train_loader, test_loader, device, is_multilabel=True):
    """训练分类器"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() if is_multilabel else nn.CrossEntropyLoss()

    best_acc = 0
    patience = 10
    no_improve = 0

    for epoch in range(100):  # 最多100个epoch
        # 训练阶段
        model.train()
        train_losses = []
        train_accs = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for batch in pbar:
            x, heart_labels, id_labels = [item.to(device) for item in batch]
            target = heart_labels if is_multilabel else id_labels.squeeze()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 计算准确率
            acc = compute_accuracy(output, target, is_multilabel)
            train_losses.append(loss.item())
            train_accs.append(acc)

            pbar.set_postfix({'loss': f"{np.mean(train_losses):.3f}",
                              'acc': f"{np.mean(train_accs):.3f}"})

        # 验证阶段
        model.eval()
        val_accs = []
        with torch.no_grad():
            for batch in test_loader:
                x, heart_labels, id_labels = [item.to(device) for item in batch]
                target = heart_labels if is_multilabel else id_labels.squeeze()

                output = model(x)
                acc = compute_accuracy(output, target, is_multilabel)
                val_accs.append(acc)

        val_acc = np.mean(val_accs)
        logger.info(f"Validation accuracy: {val_acc:.3f}")

        # 检查是否达到目标准确率
        if val_acc >= 0.98:
            logger.info("Reached target accuracy of 98%!")
            return model

        # 检查是否需要早停
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(),
                       'heart_model.pth' if is_multilabel else 'id_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping!")
                break

    if best_acc < 0.98:
        logger.warning("Failed to reach target accuracy of 98%")

    # 加载最佳模型
    model.load_state_dict(torch.load(
        'heart_model.pth' if is_multilabel else 'id_model.pth'))
    return model


import matplotlib.pyplot as plt


def plot_processed_vs_original(test_loader, processor, device, num_samples=50):
    """
    绘制原始信号与处理后信号的对比图
    Args:
        test_loader: 测试数据加载器
        processor: 隐私处理模型
        device: 设备（CPU/GPU）
        num_samples: 绘制的样本数量
    """
    processor.eval()
    raw_signals = []
    processed_signals = []

    # 提取前 num_samples 个信号
    with torch.no_grad():
        for batch_idx, (x_raw, _, _) in enumerate(test_loader):
            if len(raw_signals) >= num_samples:
                break
            x_raw = x_raw.to(device)
            x_processed = processor(x_raw) + x_raw
            raw_signals.extend(x_raw.squeeze(1).cpu().numpy())
            processed_signals.extend(x_processed.squeeze(1).cpu().numpy())

    # 绘制
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 2))
    for i in range(num_samples):
        # 绘制原始信号
        axes[i, 0].plot(raw_signals[i])
        axes[i, 0].set_title(f"Original Signal {i + 1}")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].set_xlabel("Time")

        # 绘制处理信号
        axes[i, 1].plot(processed_signals[i])
        axes[i, 1].set_title(f"Processed Signal {i + 1}")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].set_xlabel("Time")

    plt.tight_layout()
    plt.savefig("processed_vs_original.png")


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 准备数据
    dataset = ECGDataset('../data/raw/mit/')  # 自动下载和处理数据

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=4)

    # 打印信息
    print(f"Dataset size: {len(dataset)}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Test Loader batches: {len(test_loader)}")

    for signals, labels_heart, labels_id in train_loader:
        print(f"Signals batch shape: {signals.shape}")  # 打印信号的形状
        print(f"Heart labels batch shape: {labels_heart.shape}")  # 打印心跳标签的形状
        print(f"ID labels batch shape: {labels_id.shape}")  # 打印病人ID标签的形状
        break

    # 2. 训练心跳分类模型
    logger.info("Training heart beat classifier...")
    heart_model = ECGClassifier(num_classes=5, is_multilabel=True)
    heart_model = train_classifier(
        heart_model, train_loader, test_loader, device, is_multilabel=True)

    # 3. 训练ID识别模型
    logger.info("Training ID classifier...")
    id_model = ECGClassifier(num_classes=len(dataset.id_to_idx), is_multilabel=False)
    id_model = train_classifier(
        id_model, train_loader, test_loader, device, is_multilabel=False)

    # # 频段重要性分析
    # freq_importance_id, freq_edges = compute_frequency_importance(
    #     id_model, train_loader, fs=360, freq_bins=360, device=device
    # )
    # freq_importance_heart, _ = compute_frequency_importance(
    #     heart_model, train_loader, fs=360, freq_bins=360, device=device
    # )
    #
    # # 绘制频率重要性热力图
    # plot_frequency_heatmap(freq_importance_id, freq_importance_heart, freq_edges)
    #
    # # 生成频段指导掩模
    # freq_guidance = compute_freq_guidance_mask(freq_importance_id, freq_importance_heart, device)
    # print(freq_guidance)

    # # 初始化隐私处理器
    # processor = ECGPrivacyProcessor(
    #     input_dim=3600,
    #     freq_guidance=freq_guidance,
    #     noise_limit=1.5,
    #     device=device
    # ).to(device)

    # # 初始化训练器
    # trainer = PrivacyProtectionTrainer(
    #     processor=processor,
    #     heart_model=heart_model,
    #     id_model=id_model,
    #     device=device

    # 替换原有的处理器
    processor = DiffusionPrivacyProcessor(
        signal_length=3600,  # ECG信号长度
        time_steps=1000  # 扩散步数
    )

    trainer = PrivacyProtectionTrainer(
        processor=processor,
        heart_model=heart_model,
        id_model=id_model,
        device=device
    )
    # )

    # # 4. 初始化处理器和训练器
    # processor = ECGPrivacyProcessor()
    # trainer = PrivacyProtectionTrainer(
    #     processor=processor,
    #     heart_model=heart_model,
    #     id_model=id_model,
    #     device=device
    # )

    # 5. 训练处理模型
    best_metrics = trainer.train(train_loader, test_loader)
    logger.info(f"Training completed with best metrics: {best_metrics}")

    # 调用函数
    plot_processed_vs_original(test_loader, processor, device, num_samples=50)


if __name__ == '__main__':
    main()