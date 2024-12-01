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
    data_path = './data/raw/mit/'
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
            'Q': 4   # 未知类型
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
        extended_end = end - half_beat      # 确保最后一个心跳完整
        
        # 获取扩展窗口范围内的所有R波
        for i, peak in enumerate(annotations['r_peaks']):
            # 只包含完整的心跳
            if extended_start <= peak < extended_end:
                # 检查前后是否有足够空间容纳完整心跳
                prev_peak = annotations['r_peaks'][i-1] if i > 0 else peak - self.p_qrs_t_duration
                next_peak = annotations['r_peaks'][i+1] if i < len(annotations['r_peaks'])-1 else peak + self.p_qrs_t_duration
                
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
            if r_peaks[i] - r_peaks[i-1] < self.p_qrs_t_duration:
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

class ECGPrivacyProcessor(nn.Module):
    """ECG隐私处理模型"""
    def __init__(self):
       super().__init__()
       
       # 特征提取器
       self.feature_extractor = nn.Sequential(
           nn.Conv1d(1, 16, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Conv1d(16, 32, kernel_size=3, padding=1),
           nn.ReLU(),
       )
       
       # 变换参数生成网络 - 使用tanh替代sigmoid
       self.transform_params = nn.Sequential(
           nn.Conv1d(32, 16, kernel_size=1),
           nn.ReLU(),
           nn.Conv1d(16, 3, kernel_size=1),
           nn.Tanh()  # 允许正负变化
       )
       
       # 可学习的变换强度
       self.amp_strength = nn.Parameter(torch.tensor(1.0))
       self.phase_strength = nn.Parameter(torch.tensor(1.0))
       self.morph_strength = nn.Parameter(torch.tensor(1.0))
   
    def forward(self, x):
       # 提取特征
       features = self.feature_extractor(x)
       
       # 生成变换参数
       params = self.transform_params(features)
       
       # 应用变换 - 增加变换强度
       amp_factor = params[:, 0:1, :] * self.amp_strength
       x_amp = x * (1 + amp_factor)
       
       phase_factor = params[:, 1:2, :] * math.pi * self.phase_strength
       x_phase = self._apply_phase_modulation(x, phase_factor)
       
       morph_factor = params[:, 2:3, :] * self.morph_strength
       x_morph = x + morph_factor * self._get_morph_transform(x)
       
       # 组合变换 - 添加残差连接
       x_transformed = x + (x_amp - x) + (x_phase - x) + (x_morph - x)
       return x_transformed
    
    def _apply_phase_modulation(self, x, phase):
        """应用相位调制"""
        # 转到频域
        fft = torch.fft.rfft(x, dim=2)
        
        # 调整phase维度以匹配fft
        phase = F.interpolate(phase, size=fft.shape[2], mode='linear')
        
        # 应用相位调制
        modulated_fft = fft * torch.exp(1j * phase)
        
        # 转回时域
        return torch.fft.irfft(modulated_fft, n=x.shape[2], dim=2)
    
    def _get_morph_transform(self, x):
        """获取形态学变换mask"""
        # 在batch上循处理
        batch_size = x.shape[0]
        length = x.shape[2]
        morph_mask = torch.zeros_like(x)
        
        for i in range(batch_size):
            # 转到CPU计算peaks
            signal = x[i, 0].cpu().numpy()
            peaks, _ = find_peaks(signal)
            valleys, _ = find_peaks(-signal)
            
            # 创建变换mask
            mask = np.zeros_like(signal)
            window = 10  # 固定窗口大小
            
            for p in peaks:
                start = max(0, p - window)
                end = min(length, p + window)
                mask[start:end] += 1
                
            for v in valleys:
                start = max(0, v - window)
                end = min(length, v + window)
                mask[start:end] -= 1
            
            morph_mask[i, 0] = torch.from_numpy(mask).to(x.device)
        
        return morph_mask

class ECGPrivacyDataset(Dataset):
    """带有隐私处理的ECG数据集"""
    def __init__(self, original_dataset, privacy_processor=None):
        self.original_dataset = original_dataset
        self.privacy_processor = privacy_processor
        self.preprocessor = ECGProcessor()
    
    def __getitem__(self, idx):
        # 取原始数据
        x_raw, heart_label, id_label = self.original_dataset[idx]
        
        # 如果有隐私处理器，先进行隐私处理
        if self.privacy_processor is not None:
            with torch.no_grad():
                x_processed = self.privacy_processor(x_raw.unsqueeze(0))
                x_processed = x_processed.squeeze(0)
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
        self.device = device
        self.processor = processor.to(device)
        self.heart_model = heart_model.to(device)
        self.id_model = id_model.to(device)
        self.preprocessor = ECGProcessor()
        
        # 调整初始权重和范围
        self.lambda_heart = 15.0  # 适度提高初始值
        self.lambda_id = 0.5
        self.lambda_sim = 10.0    # 显著提高初始值
        
        # 修改权重范围
        self.max_lambda_heart = 40.0  # 提高上限
        self.min_lambda_sim = 5.0     # 提高下限

        # 目标阈值
        self.target_heart_acc = 0.95
        self.target_similarity = 0.90
        
        # 相似度计算的权重
        self.w_pcc = 0.4  # 皮尔逊相关系数权重
        self.w_rmse = 0.3 # 均方根误差权重  
        self.w_cc = 0.3   # 互相关系数权重
        
        # 优化器参数
        self.optimizer = optim.Adam(
            processor.parameters(),
            lr=0.001,  # 增加学习率以加快收敛
            weight_decay=1e-4  # 增加权重衰减
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,  # 减少patience加快调整
            verbose=True
        )
    
    def _adjust_weights(self, heart_acc, similarity):
        """修改权重调整策略"""
        if heart_acc < self.target_heart_acc:
            # 更温和地调整heart权重
            self.lambda_heart = min(self.max_lambda_heart, self.lambda_heart * 1.2)
            self.lambda_id = max(0.1, self.lambda_id * 0.8)
            # 保持较高的相似度权重
            self.lambda_sim = max(self.min_lambda_sim, self.lambda_sim)
            
        elif similarity < self.target_similarity:
            # 大幅提高相似度权重
            self.lambda_sim = min(20.0, self.lambda_sim * 1.5)
            # 保持较高的heart权重
            self.lambda_heart = max(10.0, self.lambda_heart * 0.95)
            self.lambda_id = max(0.1, self.lambda_id * 0.8)
            
        else:
            # 微调权重
            self.lambda_heart = max(10.0, self.lambda_heart * 0.98)
            self.lambda_sim = max(self.min_lambda_sim, self.lambda_sim * 0.98)
            self.lambda_id = min(2.0, self.lambda_id * 1.2)
    
    def _check_targets_met(self, metrics):
        """检查是否满足目标要求"""
        return (metrics['heart_acc'] >= self.target_heart_acc and 
                metrics['similarity'] >= self.target_similarity)
    
    def compute_similarity(self, x_raw, x_processed):
        """计算波形相似度指标"""
        # 保存原始设备信息
        device = x_raw.device
        
        # 使用detach()分离梯度
        x_raw = x_raw.detach().cpu().numpy()
        x_processed = x_processed.detach().cpu().numpy()
        
        batch_size = x_raw.shape[0]
        similarities = []
        
        for i in range(batch_size):
            # 获取单个波形
            raw = x_raw[i, 0]
            proc = x_processed[i, 0]
            
            # 1. 计算皮尔逊相关系数
            pcc, _ = pearsonr(raw, proc)
            sim_pcc = (pcc + 1) / 2  # 转换到[0,1]范围
            
            # 2. 计算归一化RMSE
            rmse = np.sqrt(np.mean((raw - proc) ** 2))
            max_rmse = np.sqrt(np.mean(raw ** 2) + np.mean(proc ** 2))
            sim_rmse = 1 - (rmse / max_rmse)  # 转换为相似度
            
            # 3. 计算互相关系数
            cc = np.corrcoef(raw, proc)[0, 1]
            sim_cc = (cc + 1) / 2  # 转换到[0,1]范围
            
            # 加权平均
            similarity = (
                self.w_pcc * sim_pcc + 
                self.w_rmse * sim_rmse + 
                self.w_cc * sim_cc
            )
            similarities.append(similarity)
        
        # 转换回tensor并移动到正确的设备
        return torch.tensor(similarities).mean().to(device)
    
    def train(self, train_loader, val_loader, epochs=150, patience=30):  # 增加训练轮数和耐心值
        """训练隐私保护处理器"""
        # 初始化最佳指标和分数
        best_metrics = None
        best_overall_score = float('-inf')  # 初始化为负无穷
        patience_counter = 0
        
        logger.info("Starting privacy protection training...")
        logger.info(f"Targets: heart_acc >= {self.target_heart_acc*100}%, "
                   f"similarity >= {self.target_similarity}, minimize id_acc")
        
        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_metrics['heart_acc'])
            
            # 动态调整权重
            self._adjust_weights(val_metrics['heart_acc'], val_metrics['similarity'])
            
            # 计算综合得分
            # 只有在满足heart_acc和similarity目标时才考虑id_acc
            if self._check_targets_met(val_metrics):
                overall_score = (val_metrics['heart_acc'] + 
                               val_metrics['similarity'] - 
                               val_metrics['id_acc'])
            else:
                overall_score = float('-inf')
            
            # 日志输出
            log_msg = (
                f"Epoch {epoch+1}/{epochs} | "
                f"Heart Acc: {val_metrics['heart_acc']*100:.1f}% | "
                f"ID Acc: {val_metrics['id_acc']*100:.1f}% | "
                f"Similarity: {val_metrics['similarity']:.3f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"λ_heart: {self.lambda_heart:.2f}, λ_id: {self.lambda_id:.2f}, "
                f"λ_sim: {self.lambda_sim:.2f}"
            )
            logger.info(log_msg)
            
            # 保存最佳模型
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_metrics = val_metrics.copy()  # 创建副本以避免引用问题
                patience_counter = 0
                torch.save(self.processor.state_dict(), 'best_privacy_processor.pth')
            else:
                patience_counter += 1
            
            # 早停条件：
            # 1. 达到耐心值
            # 2. 或者已经满足所有目标且ID准确率已经很低
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            elif (self._check_targets_met(val_metrics) and 
                  val_metrics['id_acc'] < 0.2):  # ID准确率低于20%
                logger.info("Achieved all targets with low ID accuracy!")
                break
        
        # 加载最佳模型
        self.processor.load_state_dict(torch.load('best_privacy_processor.pth'))
        return best_metrics
    
    def train_step(self, batch):
        self.processor.train()
        self.heart_model.eval()
        self.id_model.eval()
        
        # 冻结分类器参数
        for param in self.heart_model.parameters():
            param.requires_grad = False
        for param in self.id_model.parameters():
            param.requires_grad = False
        
        x_raw, heart_labels, id_labels = [item.to(self.device) for item in batch]
        id_labels = id_labels.squeeze()
        
        # 隐私处理
        x_processed = self.processor(x_raw)
        
        # 预处理 - 使用自定义函数保持梯度
        x_preprocessed = torch.stack([
            PreprocessorFunction.apply(x, self.preprocessor)
            for x in x_processed
        ])
        
        # 模型预测
        heart_pred = self.heart_model(x_preprocessed)
        id_pred = self.id_model(x_preprocessed)
        
        # 损失计算 - 使用新的权重
        heart_loss = F.binary_cross_entropy_with_logits(heart_pred, heart_labels)
        id_loss = F.cross_entropy(id_pred, id_labels)
        
        # 计算相似度损失
        similarity = self.compute_similarity(x_raw, x_processed)
        sim_loss = 1 - similarity  # 转换为损失
        
        # 总损失加入相似度项
        total_loss = (
            self.lambda_heart * heart_loss - 
            self.lambda_id * id_loss +
            self.lambda_sim * sim_loss
        )
        
        # 优化器步骤
        self.optimizer.zero_grad()
        total_loss.backward()
        # 增加梯度裁剪的阈值
        torch.nn.utils.clip_grad_norm_(self.processor.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        # 计算指标
        heart_acc = (heart_pred > 0.5).float().eq(heart_labels).float().mean()
        id_acc = (id_pred.argmax(1) == id_labels).float().mean()
        
        # 在返回的metrics中加入相似度
        metrics = {
            'total_loss': total_loss.item(),
            'heart_loss': heart_loss.item(),
            'id_loss': id_loss.item(),
            'heart_acc': heart_acc.item(),
            'id_acc': id_acc.item(),
            'similarity': similarity.item()
        }
        
        return metrics
    
    def _train_epoch(self, train_loader):
        self.processor.train()
        metrics_list = []
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            metrics = self.train_step(batch)
            metrics_list.append(metrics)
        
        return {
            k: np.mean([m[k] for m in metrics_list])
            for k in metrics_list[0].keys()
        }
    
    def _validate(self, val_loader):
        """验证模型性能"""
        self.processor.eval()
        metrics_list = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x_raw, heart_labels, id_labels = [item.to(self.device) for item in batch]
                id_labels = id_labels.squeeze()
                
                # 隐私处理
                x_processed = self.processor(x_raw)
                
                # 预处理 - 使用相同的预处理函数
                x_preprocessed = torch.stack([
                    PreprocessorFunction.apply(x, self.preprocessor)
                    for x in x_processed
                ])
                
                # 模型预测
                heart_pred = self.heart_model(x_preprocessed)
                id_pred = self.id_model(x_preprocessed)
                
                # 计算损失
                heart_loss = F.binary_cross_entropy_with_logits(heart_pred, heart_labels)
                id_loss = F.cross_entropy(id_pred, id_labels)
                
                # 计算相似度损失
                similarity = self.compute_similarity(x_raw, x_processed)
                sim_loss = 1 - similarity  # 转换为损失
                
                # 总损失加入相似度项
                total_loss = (
                    self.lambda_heart * heart_loss - 
                    self.lambda_id * id_loss +
                    self.lambda_sim * sim_loss
                )
                
                # 计算指标
                heart_acc = (heart_pred > 0.5).float().eq(heart_labels).float().mean()
                id_acc = (id_pred.argmax(1) == id_labels).float().mean()
                
                metrics_list.append({
                    'total_loss': total_loss.item(),
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
    
    def protect_batch(self, batch):
        """对批次数据进行隐私保护处理并评估效果"""
        self.processor.eval()
        with torch.no_grad():
            x_raw, heart_labels, id_labels = [item.to(self.device) for item in batch]
            id_labels = id_labels.squeeze()
            
            # 隐私处理
            x_processed = self.processor(x_raw)
            
            # 预处理 - 使用相同的预处理函数
            x_preprocessed = torch.stack([
                PreprocessorFunction.apply(x, self.preprocessor)
                for x in x_processed
            ])
            
            # 模型预测
            heart_pred = self.heart_model(x_preprocessed)
            id_pred = self.id_model(x_preprocessed)
            
            # 计算损失
            heart_loss = F.binary_cross_entropy_with_logits(heart_pred, heart_labels)
            id_loss = F.cross_entropy(id_pred, id_labels)
            
            # 计算相似度损失
            similarity = self.compute_similarity(x_raw, x_processed)
            sim_loss = 1 - similarity  # 转换为损失
            
            # 总损失加入相似度项
            total_loss = (
                self.lambda_heart * heart_loss - 
                self.lambda_id * id_loss +
                self.lambda_sim * sim_loss
            )
            
            # 计算指标
            heart_acc = (heart_pred > 0.5).float().eq(heart_labels).float().mean()
            id_acc = (id_pred.argmax(1) == id_labels).float().mean()
            
            metrics = {
                'total_loss': total_loss.item(),
                'heart_loss': heart_loss.item(),
                'id_loss': id_loss.item(),
                'heart_acc': heart_acc.item(),
                'id_acc': id_acc.item(),
                'similarity': similarity.item()
            }
            
            return x_processed, metrics

def compute_accuracy(output, target, is_multilabel):
    """计算准确率"""
    if is_multilabel:
        pred = (output > 0.5).float()
        correct = (pred == target).float().mean()
    else:
        pred = output.argmax(dim=1)
        correct = (pred == target).float().mean()
    return correct.item()

def train_classifier(model, train_loader, val_loader, device, is_multilabel=True):
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
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
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
            for batch in val_loader:
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

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 准备数据
    dataset = ECGDataset('./data/raw/mit/')  # 自动下载和处理数据
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                            num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=4)
    
    # 2. 训练心跳分类模型
    logger.info("Training heart beat classifier...")
    heart_model = ECGClassifier(num_classes=5, is_multilabel=True)
    heart_model = train_classifier(
        heart_model, train_loader, val_loader, device, is_multilabel=True)
    
    # 3. 训练ID识别模型
    logger.info("Training ID classifier...")
    id_model = ECGClassifier(num_classes=len(dataset.id_to_idx), is_multilabel=False)
    id_model = train_classifier(
        id_model, train_loader, val_loader, device, is_multilabel=False)
    
    # 4. 初始化处理器和训练器
    processor = ECGPrivacyProcessor()
    trainer = PrivacyProtectionTrainer(
        processor=processor,
        heart_model=heart_model,
        id_model=id_model,
        device=device
    )
    
    # 5. 训练处理模型
    best_metrics = trainer.train(train_loader, val_loader)
    logger.info(f"Training completed with best metrics: {best_metrics}")

if __name__ == '__main__':
    main()