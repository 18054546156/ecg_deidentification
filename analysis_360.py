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
from PyEMD import EMD
from sklearn.cluster import KMeans
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import scipy.stats
import pywt


class ECGAnalyzer:
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate

    def analyze_and_save(self, signals, heart_labels, id_labels, save_dir='analysis_results'):
        """分析ECG数据并保存结果"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 1. 频域分析
        print("进行频域分析...")
        freq_results = self._analyze_frequency(signals, heart_labels, id_labels)
        self._save_frequency_plots(freq_results, save_dir)

        # 2. 时域分析
        print("进行时域分析...")
        time_results = self._analyze_time(signals, heart_labels, id_labels)
        self._save_time_plots(time_results, save_dir)

        # 3. 机器学习分析
        print("进行特征分析...")
        ml_results = self._analyze_ml_features(signals, heart_labels, id_labels)
        self._save_ml_plots(ml_results, save_dir)

        # 4. 保存分析报告
        print("生成分析报告...")
        self._save_report({
            'frequency': freq_results,
            'time': time_results,
            'ml': ml_results
        }, save_dir)

        return {
            'frequency': freq_results,
            'time': time_results,
            'ml': ml_results
        }

    def _analyze_frequency(self, signals, heart_labels, id_labels):
        """频域分析"""
        freq_results = {}

        # 1. 计算功率谱密度
        freqs, psd = signal.welch(signals, fs=self.sampling_rate,
                                  nperseg=min(signals.shape[1], 1024),
                                  scaling='density')
        freq_results['freqs'] = freqs
        freq_results['psd'] = psd

        # 2. 按心跳类别分组的PSD
        unique_heart = np.unique(heart_labels)
        heart_psd = np.zeros((len(unique_heart), len(freqs)))
        for i, label in enumerate(unique_heart):
            mask = heart_labels == label
            heart_psd[i] = np.mean(psd[mask], axis=0)
        freq_results['heart_psd'] = heart_psd

        # 3. 按ID分组的PSD
        unique_id = np.unique(id_labels)
        id_psd = np.zeros((len(unique_id), len(freqs)))
        for i, label in enumerate(unique_id):
            mask = id_labels == label
            id_psd[i] = np.mean(psd[mask], axis=0)
        freq_results['id_psd'] = id_psd

        # 4. 频段分析
        bands = {
            'vlf': (0, 0.5),  # 极低频
            'lf': (0.5, 10),  # 低频
            'mf': (10, 50),  # 中频
            'hf': (50, 100),  # 高频
            'vhf': (100, 250)  # 极高频
        }

        # 5. 计算每个频段的能量
        band_energy = {}
        band_discrimination = {}

        for band_name, (low, high) in bands.items():
            # 找到频段对应的索引
            mask = (freqs >= low) & (freqs < high)
            band_psd = psd[:, mask]

            # 计算每个样本在该频段的总能量
            energy = np.sum(band_psd, axis=1)

            # 分别计算心跳任务和ID任务的区分度
            # 心跳任务的区分度
            heart_energies = {}
            for label in unique_heart:
                mask = heart_labels == label
                heart_energies[label] = energy[mask]

            heart_discrim = self._calculate_discrimination(heart_energies)

            # ID任务的区分度
            id_energies = {}
            for label in unique_id:
                mask = id_labels == label
                id_energies[label] = energy[mask]

            id_discrim = self._calculate_discrimination(id_energies)

            # 保存结果
            band_energy[band_name] = {
                'heart': heart_energies,
                'id': id_energies
            }

            band_discrimination[band_name] = {
                'heart': heart_discrim,
                'id': id_discrim,
                'ratio': id_discrim / (heart_discrim + 1e-10)  # 避免除零
            }

        freq_results['band_energy'] = band_energy
        freq_results['band_discrimination'] = band_discrimination

        # 6. 小波分析
        wavelet_results = self._wavelet_analysis(signals)
        freq_results['wavelet'] = wavelet_results

        return freq_results

    def _calculate_discrimination(self, group_values):
        """计算组间区分度（Fisher判别比）"""
        # 计算组内均值
        means = np.array([np.mean(values) for values in group_values.values()])

        # 计算组内方差
        vars = np.array([np.var(values) for values in group_values.values()])

        # 计算总体均值
        total_mean = np.mean(means)

        # 计算组间方差
        between_var = np.mean((means - total_mean) ** 2)

        # 计算平均组内方差
        within_var = np.mean(vars)

        # 计算Fisher判别比
        fisher_ratio = between_var / (within_var + 1e-10)

        return fisher_ratio

    def _wavelet_analysis(self, signals):
        """小波分析"""
        # 使用连续小波变换分析时频特征
        wavelet = 'morl'  # Morlet小波
        scales = np.arange(1, 128)  # 尺度范围

        # 对每个信号进行小波变换
        coefficients = []
        frequencies = pywt.scale2frequency(wavelet, scales) * self.sampling_rate

        # 只分析部分信号以节省时间
        n_samples = min(100, len(signals))
        for signal in signals[:n_samples]:
            coef, _ = pywt.cwt(signal, scales, wavelet, 1 / self.sampling_rate)
            coefficients.append(coef)

        return {
            'coefficients': np.array(coefficients),
            'frequencies': frequencies,
            'scales': scales
        }

    def _save_frequency_plots(self, freq_results, save_dir):
        """保存频域分析图"""
        freq_dir = os.path.join(save_dir, 'frequency')
        os.makedirs(freq_dir, exist_ok=True)

        # 1. PSD图
        plt.figure(figsize=(12, 6))
        for i, psd in enumerate(freq_results['heart_psd']):
            plt.plot(freq_results['freqs'], psd, alpha=0.5,
                     label=f'Heart Class {i}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.title('Power Spectral Density by Heart Classes')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(freq_dir, 'psd_heart.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 频段区分度对比图
        plt.figure(figsize=(10, 6))
        bands = list(freq_results['band_discrimination'].keys())
        x = np.arange(len(bands))
        width = 0.35

        heart_discrim = [freq_results['band_discrimination'][band]['heart']
                         for band in bands]
        id_discrim = [freq_results['band_discrimination'][band]['id']
                      for band in bands]

        plt.bar(x - width / 2, heart_discrim, width, label='Heart Task')
        plt.bar(x + width / 2, id_discrim, width, label='ID Task')
        plt.xticks(x, bands, rotation=45)
        plt.title('Frequency Band Discrimination')
        plt.ylabel('Fisher Ratio')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(freq_dir, 'band_discrimination.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 时频图（小波分析）
        if 'wavelet' in freq_results:
            plt.figure(figsize=(12, 8))
            plt.imshow(np.mean(np.abs(freq_results['wavelet']['coefficients']), axis=0),
                       aspect='auto', cmap='jet',
                       extent=[0, signals.shape[1] / self.sampling_rate,
                               freq_results['wavelet']['frequencies'][-1],
                               freq_results['wavelet']['frequencies'][0]])
            plt.colorbar(label='Magnitude')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            plt.title('Average Wavelet Scalogram')
            plt.savefig(os.path.join(freq_dir, 'wavelet_scalogram.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

    def _analyze_time(self, signals, heart_labels, id_labels):
        """时域分析"""
        time_results = {}

        # 1. R峰检测
        r_peaks = []
        r_peak_values = []
        for signal in signals:
            # 使用Pan-Tompkins算法的简化版本
            # 带通滤波
            b, a = signal.butter(3, [5 / self.sampling_rate * 2, 15 / self.sampling_rate * 2], 'bandpass')
            filtered = signal.filtfilt(b, a, signal)

            # 求导
            diff = np.diff(filtered)

            # 平方
            squared = diff ** 2

            # 移动平均
            window_size = int(0.15 * self.sampling_rate)  # 150ms窗口
            window = np.ones(window_size) / window_size
            smoothed = np.convolve(squared, window, mode='same')

            # 找峰值
            peaks, _ = signal.find_peaks(smoothed,
                                         height=np.mean(smoothed),
                                         distance=0.2 * self.sampling_rate)

            r_peaks.append(peaks)
            r_peak_values.append(signal[peaks])

        time_results['r_peaks'] = r_peaks
        time_results['r_peak_values'] = r_peak_values

        # 2. RR间期分析
        rr_intervals = []
        for peaks in r_peaks:
            intervals = np.diff(peaks) / self.sampling_rate  # 转换为秒
            rr_intervals.append(intervals)

        time_results['rr_intervals'] = rr_intervals

        # 3. 心率变异性分析
        hrv_features = []
        for intervals in rr_intervals:
            if len(intervals) > 1:
                # 时域HRV特征
                hrv = {
                    'mean_rr': np.mean(intervals),
                    'std_rr': np.std(intervals),
                    'rmssd': np.sqrt(np.mean(np.diff(intervals) ** 2)),  # 相邻RR间期差值的均方根
                    'pnn50': np.sum(np.abs(np.diff(intervals)) > 0.05) / len(intervals) * 100  # pNN50
                }
                hrv_features.append(hrv)

        time_results['hrv_features'] = hrv_features

        # 4. 波形形态分析
        morphology_features = []
        for signal in signals:
            # 计算波形统计特征
            morph = {
                'mean': np.mean(signal),
                'std': np.std(signal),
                'skewness': scipy.stats.skew(signal),
                'kurtosis': scipy.stats.kurtosis(signal),
                'peak_to_peak': np.max(signal) - np.min(signal)
            }
            morphology_features.append(morph)

        time_results['morphology'] = morphology_features

        # 5. 按任务分组分析
        def group_analysis(features, labels):
            unique_labels = np.unique(labels)
            grouped_features = {}
            for label in unique_labels:
                mask = labels == label
                if isinstance(features[0], dict):
                    # 对于字典类型的特征
                    grouped_features[label] = {
                        k: np.mean([f[k] for f in features[mask]])
                        for k in features[0].keys()
                    }
                else:
                    # 对于数组类型的特征
                    grouped_features[label] = np.mean(features[mask], axis=0)
            return grouped_features

        # 分组分析HRV特征
        time_results['grouped_hrv'] = {
            'heart': group_analysis(hrv_features, heart_labels),
            'id': group_analysis(hrv_features, id_labels)
        }

        # 分组分析形态特征
        time_results['grouped_morphology'] = {
            'heart': group_analysis(morphology_features, heart_labels),
            'id': group_analysis(morphology_features, id_labels)
        }

        return time_results

    def _save_time_plots(self, time_results, save_dir):
        """保存时域分析图"""
        time_dir = os.path.join(save_dir, 'time')
        os.makedirs(time_dir, exist_ok=True)

        # 1. RR间期分布图
        plt.figure(figsize=(10, 6))
        all_intervals = np.concatenate(time_results['rr_intervals'])
        plt.hist(all_intervals, bins=50, alpha=0.75)
        plt.title('RR Interval Distribution')
        plt.xlabel('RR Interval (s)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(time_dir, 'rr_intervals.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. HRV特征箱线图
        plt.figure(figsize=(12, 6))
        hrv_data = {
            'MEAN_RR': [f['mean_rr'] for f in time_results['hrv_features']],
            'STD_RR': [f['std_rr'] for f in time_results['hrv_features']],
            'RMSSD': [f['rmssd'] for f in time_results['hrv_features']],
            'pNN50': [f['pnn50'] for f in time_results['hrv_features']]
        }
        plt.boxplot(list(hrv_data.values()), labels=list(hrv_data.keys()))
        plt.title('HRV Features Distribution')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(time_dir, 'hrv_features.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 形态特征分布图
        plt.figure(figsize=(12, 6))
        morph_data = {
            'Mean': [f['mean'] for f in time_results['morphology']],
            'STD': [f['std'] for f in time_results['morphology']],
            'Skewness': [f['skewness'] for f in time_results['morphology']],
            'Kurtosis': [f['kurtosis'] for f in time_results['morphology']],
            'P2P': [f['peak_to_peak'] for f in time_results['morphology']]
        }
        plt.boxplot(list(morph_data.values()), labels=list(morph_data.keys()))
        plt.title('Morphology Features Distribution')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(time_dir, 'morphology_features.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_ml_features(self, signals, heart_labels, id_labels):
        """机器学习特征分析"""
        # 提取特征
        features = self._extract_features(signals)

        # 随机森林特征重要性
        rf_heart = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_id = RandomForestClassifier(n_estimators=100, random_state=42)

        rf_heart.fit(features, heart_labels)
        rf_id.fit(features, id_labels)

        # 互信息分析
        mi_heart = mutual_info_classif(features, heart_labels)
        mi_id = mutual_info_classif(features, id_labels)

        # PCA分析
        pca = PCA()
        pca.fit(features)

        return {
            'feature_importance': {
                'heart': rf_heart.feature_importances_,
                'id': rf_id.feature_importances_
            },
            'mutual_information': {
                'heart': mi_heart,
                'id': mi_id
            },
            'pca': {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'components': pca.components_
            }
        }

    def _save_ml_plots(self, ml_results, save_dir):
        """保存机器学习分析图"""
        ml_dir = os.path.join(save_dir, 'ml')
        os.makedirs(ml_dir, exist_ok=True)

        # 特征重要性对比图
        plt.figure(figsize=(12, 6))
        x = np.arange(len(ml_results['feature_importance']['heart']))
        width = 0.35
        plt.bar(x - width / 2, ml_results['feature_importance']['heart'],
                width, label='Heart Task')
        plt.bar(x + width / 2, ml_results['feature_importance']['id'],
                width, label='ID Task')
        plt.title('Feature Importance Comparison')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.legend()
        plt.savefig(os.path.join(ml_dir, 'feature_importance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # PCA解释方差图
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(ml_results['pca']['explained_variance_ratio']))
        plt.title('PCA Explained Variance Ratio')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.savefig(os.path.join(ml_dir, 'pca_variance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _save_report(self, results, save_dir):
        """保存分析报告"""
        report_path = os.path.join(save_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== ECG数据分析报告 ===\n\n")

            # 频域分析结果
            f.write("1. 频域分析结果:\n")
            f.write("-----------------\n")
            for band, discrim in results['frequency']['band_discrimination'].items():
                f.write(f"{band}频段的区分度: {discrim:.4f}\n")
            f.write("\n")

            # 特征重要性结果
            f.write("2. 特��重要性分析:\n")
            f.write("------------------\n")
            heart_imp = results['ml']['feature_importance']['heart']
            id_imp = results['ml']['feature_importance']['id']

            f.write("心跳任务top3特征:\n")
            top3_heart = np.argsort(heart_imp)[-3:]
            for idx in top3_heart:
                f.write(f"特征{idx}: {heart_imp[idx]:.4f}\n")

            f.write("\nID任务top3特征:\n")
            top3_id = np.argsort(id_imp)[-3:]
            for idx in top3_id:
                f.write(f"特征{idx}: {id_imp[idx]:.4f}\n")

    def _extract_features(self, signals):
        """特征提取"""
        features = []
        for signal in signals:
            # 时域特征
            time_features = [
                np.mean(signal),
                np.std(signal),
                np.max(signal),
                np.min(signal),
                np.median(signal)
            ]

            # 频域特征
            freqs, psd = signal.welch(signal, fs=self.sampling_rate)
            freq_features = [
                np.sum(psd),
                np.mean(psd),
                np.std(psd),
                np.max(psd)
            ]

            features.append(time_features + freq_features)

        return np.array(features)


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
        # 读取标数据
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

        # 计有效的窗口始位置
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
        获取窗口内的���信息，包括完整的心跳
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
        检查窗口心跳的完整性
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
        # 1. ���线漂移校正
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
        """回元组格式的数据，与work-backup2.py保持一致"""
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


class WaveletEMDDiffusion(nn.Module):
    """结合小波变换和EMD的扩散隐私处理器"""

    def __init__(self, signal_length=3600, time_steps=1000):
        super().__init__()
        self.signal_length = signal_length
        self.time_steps = time_steps

        # 特征分解网络
        self.feature_decomposer = nn.ModuleDict({
            'wavelet': WaveletDecomposer(),  # 小波分解
            'emd': EMDDecomposer(),  # EMD分解
        })

        # 特征融合层
        self.feature_fusion_wavelet = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=1),  # 将单通道扩展为3通道
            nn.LeakyReLU(0.2)
        )

        self.feature_fusion_emd = nn.Sequential(
            nn.Conv1d(7, 3, kernel_size=1),  # 将7通道压缩为3通道
            nn.LeakyReLU(0.2)
        )

        # 扩散模型
        self.noise_predictor = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 1, kernel_size=3, padding=1)
        )

        # 生成beta schedule
        self.beta = self._linear_beta_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def _linear_beta_schedule(self):
        """生成线��beta schedule"""
        scale = 1000 / self.time_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.time_steps)

    def _extract(self, a, t, x_shape):
        """从a中提取适当的系数用于时���步t"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, 1, 1).to(t.device)

    def decompose_signal(self, x):
        """分解信号为理特征和身份特征"""
        # 小波变换分解
        wavelet_coeffs = self.feature_decomposer['wavelet'](x)
        # EMD分解
        imfs = self.feature_decomposer['emd'](x)

        # 分离特征
        physio_features = {
            'wavelet': wavelet_coeffs['physio'],  # 生理相关的小波系数
            'emd': imfs['physio']  # 生理相关的IMF
        }

        identity_features = {
            'wavelet': wavelet_coeffs['identity'],  # 身份相关的小波系数
            'emd': imfs['identity']  # 身份相关的IMF
        }

        return physio_features, identity_features

    def forward(self, x, print_info=False):
        """前向传播"""
        device = x.device
        batch_size = x.shape[0]

        # 1. 特征分解
        physio_features, identity_features = self.decompose_signal(x)

        if print_info:
            print("原始信号形状:", x.shape)
            print("生理特征:", {k: v.shape for k, v in physio_features.items()})
            print("身份特征:", {k: v.shape for k, v in identity_features.items()})

        # 2. 对身份特征应用扩散
        identity_processed = {}
        for key, feature in identity_features.items():
            # 初始化噪声
            noisy_feature = feature + 0.001 * torch.randn_like(feature)

            # 根据特征类型选择不同的融合层
            if key == 'wavelet':
                noisy_feature = self.feature_fusion_wavelet(noisy_feature)
            else:  # emd
                noisy_feature = self.feature_fusion_emd(noisy_feature)

            # 逐步扩散
            for t in reversed(range(0, self.time_steps)):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                noisy_feature = self.diffusion_step(noisy_feature, t_batch)

            # 取回单通道结果
            identity_processed[key] = noisy_feature

            if print_info:
                print(f"处理后的{key}身份���征:", noisy_feature.shape)

        # 3. 重建信号
        x_processed = self.reconstruct_signal(physio_features, identity_processed)

        if print_info:
            print("处理后的信号形状:", x_processed.shape)

        return x_processed

    def diffusion_step(self, x, t):
        """单步扩散"""
        # 预测噪声
        predicted_noise = self.noise_predictor(x)

        # 应用去噪
        alpha = self._extract(self.alpha, t, x.shape)
        alpha_hat = self._extract(self.alpha_hat, t, x.shape)
        beta = self._extract(self.beta, t, x.shape)

        noise = torch.randn_like(predicted_noise) if t[0] > 0 else 0
        mean = (x - beta * predicted_noise / torch.sqrt(1 - alpha_hat)) / torch.sqrt(alpha)

        return mean + torch.sqrt(beta) * noise

    def reconstruct_signal(self, physio_features, identity_processed):
        """重建信号"""
        # 重建wavelet部分
        reconstructed = self.feature_decomposer['wavelet'].reconstruct(
            physio_features['wavelet'],
            identity_processed['wavelet'].detach()  # 添加detach
        )

        # 重建EMD部分
        reconstructed += self.feature_decomposer['emd'].reconstruct(
            physio_features['emd'],
            identity_processed['emd'].detach()  # 添加detach
        )

        return reconstructed


class WaveletDecomposer(nn.Module):
    """小波换分解器"""

    def __init__(self):
        super().__init__()
        self.wavelet = 'db4'  # 选择小波基
        self.level = 4  # 分解层数

    def forward(self, x):
        """小波分解"""
        batch_size = x.shape[0]
        physio_features = []
        identity_features = []

        # 对每个batch的信号进行处理
        for i in range(batch_size):
            signal = x[i, 0].cpu().numpy()  # 获取单个信号
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

            # 分离生理和身份特征
            physio_coeffs = []
            identity_coeffs = []

            # 处理每个尺度的系数
            for j, coeff in enumerate(coeffs):
                if j <= 2:  # 低频部分作为生理特征
                    physio_coeffs.append(coeff)
                    identity_coeffs.append(np.zeros_like(coeff))
                else:  # 高频部分作为身份特征
                    physio_coeffs.append(np.zeros_like(coeff))
                    identity_coeffs.append(coeff)

            # 重建各自的信号
            physio_signal = pywt.waverec(physio_coeffs, self.wavelet)
            identity_signal = pywt.waverec(identity_coeffs, self.wavelet)

            # 确保长度一致
            target_length = x.shape[2]
            if len(physio_signal) > target_length:
                physio_signal = physio_signal[:target_length]
                identity_signal = identity_signal[:target_length]
            elif len(physio_signal) < target_length:
                pad_width = target_length - len(physio_signal)
                physio_signal = np.pad(physio_signal, (0, pad_width), 'constant')
                identity_signal = np.pad(identity_signal, (0, pad_width), 'constant')

            physio_features.append(physio_signal)
            identity_features.append(identity_signal)

        # 转换为tensor
        physio_features = torch.from_numpy(np.stack(physio_features)).float().to(x.device)
        identity_features = torch.from_numpy(np.stack(identity_features)).float().to(x.device)

        return {
            'physio': physio_features.unsqueeze(1),  # [B, 1, L]
            'identity': identity_features.unsqueeze(1)  # [B, 1, L]
        }

    def reconstruct(self, physio_features, identity_features):
        """重建信号"""
        # 保存原始设备信息
        device = physio_features.device

        # 确保输入是正确的形状并分离梯度
        if physio_features.dim() == 3:
            physio_features = physio_features.squeeze(1)  # [B, L]
        if identity_features.dim() == 3:
            # 如果是3通道，取平均
            identity_features = identity_features.mean(dim=1)  # [B, L]

        # 转换为numpy，分离梯度
        physio_features = physio_features.detach().cpu().numpy()
        identity_features = identity_features.detach().cpu().numpy()

        # 直接相加重建
        reconstructed = physio_features + identity_features

        # 转换回tensor并保持维度
        reconstructed = torch.from_numpy(reconstructed).float().to(device)
        return reconstructed.unsqueeze(1)  # [B, 1, L]


class EMDDecomposer(nn.Module):
    """EMD分解器"""

    def __init__(self, min_length=100, max_imf=10):
        super().__init__()
        self.min_length = min_length
        self.max_imf = max_imf
        self.emd = EMD()
        self.emd.MAX_ITERATION = 50

    def check_signal(self, signal):
        """检查信号是否适合EMD分解"""
        if len(signal) < self.min_length:
            return False
        if np.all(signal == 0):
            return False
        if np.any(np.isnan(signal)):
            return False
        if np.std(signal) < 1e-6:  # 信号过于平稳
            return False
        return True

    def safe_emd(self, signal):
        """安全的EMD分解"""
        try:
            # 添加时间序列
            t = np.linspace(0, len(signal) - 1, len(signal))
            # 使用带时间序列的EMD分解
            imfs = self.emd.emd(signal, t)

            # 限制IMF数量
            if len(imfs) > self.max_imf:
                imfs = imfs[:self.max_imf]
            elif len(imfs) < 3:  # 至少需要3个IMF
                # 如果IMF不足，用零填充
                pad_imfs = np.zeros((3 - len(imfs), len(signal)))
                imfs = np.vstack([imfs, pad_imfs])

            return imfs
        except Exception as e:
            print(f"EMD分解失败: {str(e)}")
            # 返回默认的IMF（全零）
            return np.zeros((3, len(signal)))

    def forward(self, x):
        """EMD分解"""
        batch_size = x.shape[0]
        signal_length = x.shape[2]

        # 预分配数组
        all_physio_imfs = np.zeros((batch_size, 3, signal_length))  # 3个生理IMF
        all_identity_imfs = np.zeros((batch_size, self.max_imf - 3, signal_length))  # 剩余为身份IMF

        for i in range(batch_size):
            signal = x[i, 0].cpu().numpy()

            # 检查信号
            if self.check_signal(signal):
                # EMD分解
                imfs = self.safe_emd(signal)

                # 分离生理和身份特征
                if len(imfs) >= 3:
                    all_physio_imfs[i] = imfs[:3]  # 前3个IMF作为生理特征
                    if len(imfs) > 3:
                        identity_imfs = imfs[3:]
                        # 确保身份IMF数量一致
                        if len(identity_imfs) > self.max_imf - 3:
                            identity_imfs = identity_imfs[:self.max_imf - 3]
                        all_identity_imfs[i, :len(identity_imfs)] = identity_imfs
            else:
                print(f"信号 {i} 不适合EMD分解，使用默认值")

        # 转换为tensor
        physio_imfs = torch.from_numpy(all_physio_imfs).float().to(x.device)
        identity_imfs = torch.from_numpy(all_identity_imfs).float().to(x.device)

        return {
            'physio': physio_imfs,  # [B, 3, L]
            'identity': identity_imfs  # [B, max_imf-3, L]
        }

    def reconstruct(self, physio_features, identity_features):
        """重建信号"""
        # 保存原始设备信息
        device = physio_features.device

        # 确保输入是正确的形状并分离梯度
        if identity_features.dim() == 3:
            # 如果是3通道，取平均
            identity_features = identity_features.mean(dim=1)  # [B, L]

        # 转换为numpy，分离梯度
        physio_features = physio_features.detach().cpu().numpy()
        identity_features = identity_features.detach().cpu().numpy()

        # 合并IMF并重建
        batch_size = physio_features.shape[0]
        reconstructed = np.zeros((batch_size, physio_features.shape[-1]))

        # 对于生理特征，直接相加所有IMF
        reconstructed += np.sum(physio_features, axis=1)

        # 添���身份特征
        reconstructed += identity_features

        # 转换回tensor并保持维度
        reconstructed = torch.from_numpy(reconstructed).float().to(device)
        return reconstructed.unsqueeze(1)  # [B, 1, L]


class ECGPrivacyDataset(Dataset):
    """带有隐私处的ECG数据集"""

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
    """自定义预处理数，保持梯度传播"""

    @staticmethod
    def forward(ctx, x, preprocessor):
        # 保存输入用于反向传播
        ctx.save_for_backward(x)

        # 执预处理
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
            x_preprocessed = preprocessor.preprocess_signal(x_np)
            result = torch.from_numpy(x_preprocessed).float()

        return result.to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        # 获保存的输入
        x, = ctx.saved_tensors

        # 简单传���梯度
        grad_input = grad_output.clone()

        return grad_input, None


class FrequencyAnalyzer:
    """频段分析器：通过数据分析确定不同任务的关键频段"""

    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
        self.freq_resolution = 0.1  # 频率分辨率
        self.n_bands = 50  # 初始频段划分数量

    def analyze_task_frequencies(self, signals, labels, task_type='heart'):
        """
        分析特定任务的关键频段

        Args:
            signals: 输入信号 [batch_size, 1, signal_length]
            labels: 标签
            task_type: 任务类型 ('heart' 或 'id')
        """
        # 计算每个样本的频谱
        freqs = torch.fft.rfftfreq(signals.shape[-1], d=1 / self.sampling_rate)
        ffts = torch.fft.rfft(signals, dim=-1)
        power_specs = torch.abs(ffts) ** 2

        # 按标签分组
        unique_labels = torch.unique(labels)
        group_specs = {}

        for label in unique_labels:
            mask = (labels == label)
            group_specs[label.item()] = power_specs[mask]

        # 计算组间频谱差异
        freq_importance = torch.zeros_like(freqs)

        # 使用Fisher判别比计算每个频率的重要性
        for i in range(len(freqs)):
            class_means = []
            class_vars = []

            for specs in group_specs.values():
                class_means.append(torch.mean(specs[:, :, i]))
                class_vars.append(torch.var(specs[:, :, i]))

            # 计算组间方差
            total_mean = torch.mean(torch.tensor(class_means))
            between_class_var = torch.mean(torch.tensor([(m - total_mean) ** 2 for m in class_means]))

            # 计算组内方差
            within_class_var = torch.mean(torch.tensor(class_vars))

            # Fisher判别比
            if within_class_var > 0:
                freq_importance[i] = between_class_var / within_class_var

        return freqs, freq_importance

    def find_optimal_bands(self, signals_dict, labels_dict):
        """
        找到心跳和ID任务的最优频段划分

        Args:
            signals_dict: {'heart': heart_signals, 'id': id_signals}
            labels_dict: {'heart': heart_labels, 'id': id_labels}
        """
        # 分析心跳任务的频段重要性
        heart_freqs, heart_importance = self.analyze_task_frequencies(
            signals_dict['heart'], labels_dict['heart'], 'heart')

        # 分析ID任务的频段重要性
        id_freqs, id_importance = self.analyze_task_frequencies(
            signals_dict['id'], labels_dict['id'], 'id')

        # 计算相对重要性
        total_importance = heart_importance + id_importance
        heart_relative = heart_importance / total_importance
        id_relative = id_importance / total_importance

        # 使用聚类找到自然的频段分界点
        freq_points = torch.stack([heart_relative, id_relative], dim=1)
        clusters = self._cluster_frequencies(freq_points.numpy(), n_clusters=3)

        # 确定最终的频段划分
        bands = self._determine_bands(heart_freqs.numpy(), clusters)

        return {
            'bands': bands,
            'heart_importance': heart_importance,
            'id_importance': id_importance,
            'analysis': {
                'heart_relative': heart_relative,
                'id_relative': id_relative
            }
        }

    def _cluster_frequencies(self, freq_features, n_clusters=3):
        """使用K-means聚类划分频段"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(freq_features)

    def _determine_bands(self, frequencies, clusters):
        """根据聚类结果确定频段范围"""
        bands = {}
        cluster_freqs = {}

        # 收集每个簇的频率点
        for cluster_id in np.unique(clusters):
            cluster_freqs[cluster_id] = frequencies[clusters == cluster_id]

        # 根据频率范围排序簇
        cluster_means = {k: np.mean(v) for k, v in cluster_freqs.items()}
        sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])

        # 确定频段范围
        bands['heart'] = (np.min(cluster_freqs[sorted_clusters[0][0]]),
                          np.max(cluster_freqs[sorted_clusters[0][0]]))
        bands['id'] = (np.min(cluster_freqs[sorted_clusters[1][0]]),
                       np.max(cluster_freqs[sorted_clusters[1][0]]))
        bands['noise'] = (np.min(cluster_freqs[sorted_clusters[2][0]]),
                          np.max(cluster_freqs[sorted_clusters[2][0]]))

        return bands


class PrivacyProtectionTrainer:
    def __init__(self, processor, heart_model, id_model, device):
        self.processor = processor.to(device)
        self.heart_model = heart_model.to(device)
        self.id_model = id_model.to(device)
        self.device = device

        # 频段分析器
        self.freq_analyzer = FrequencyAnalyzer()
        self.frequency_bands = None

        # 优化器
        self.optimizer = optim.Adam(self.processor.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # 损失权重
        self.lambda_heart = 5.0  # 心跳任��权重
        self.lambda_id = 1.0  # ID任务权重
        self.lambda_spectral = 2.0  # 频谱约束权重

        # 初始化最佳指标
        self.best_metrics = None
        self.best_epoch = 0

    def analyze_data_characteristics(self, train_loader):
        """分析训练数据的频段特征"""
        print("正在分析数据频段特征...")

        # 收集数据
        heart_signals = []
        heart_labels = []
        id_signals = []
        id_labels = []

        for batch_idx, (x, h_label, i_label) in enumerate(train_loader):
            if batch_idx < 100:  # 使用部分数据进行分析
                heart_signals.append(x)
                heart_labels.append(h_label)
                id_signals.append(x)
                id_labels.append(i_label)

        # 合并数据
        heart_signals = torch.cat(heart_signals, dim=0)
        heart_labels = torch.cat(heart_labels, dim=0)
        id_signals = torch.cat(id_signals, dim=0)
        id_labels = torch.cat(id_labels, dim=0)

        # 分析频段
        analysis_results = self.freq_analyzer.find_optimal_bands(
            {'heart': heart_signals, 'id': id_signals},
            {'heart': heart_labels, 'id': id_labels}
        )

        self.frequency_bands = analysis_results['bands']

        # 打印分析结果
        print("\n频段分析结果:")
        for task, (low, high) in self.frequency_bands.items():
            print(f"{task}任务主要频段: {low:.1f}Hz - {high:.1f}Hz")

        return analysis_results

    def train(self, train_loader, test_loader, epochs=5, patience=30):
        """训练函数"""
        # 首先进行数据分析
        if self.frequency_bands is None:
            analysis_results = self.analyze_data_characteristics(train_loader)
            # 更新处理器的频段设置
            self.processor.update_frequency_bands(self.frequency_bands)

        # 继续训练过程...

    def analyze_signal(self, x):
        """分析信号的频谱特征"""
        # 计算频谱
        x_fft = torch.fft.rfft(x, dim=2)
        freqs = torch.fft.rfftfreq(x.shape[2], d=1 / 500)

        # 分析各频段能量
        bands = {
            'heart': (0.5, 50),
            'id': (50, 100),
            'noise': (100, 250)
        }

        energies = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            band_energy = torch.mean(torch.abs(x_fft[:, :, mask]) ** 2)
            energies[band_name] = band_energy.item()

        return energies

    def evaluate(self, test_loader):
        """评估模型性能"""
        self.processor.eval()
        total_heart_acc = 0
        total_id_acc = 0
        num_batches = 0

        with torch.no_grad():
            for x, heart_labels, id_labels in test_loader:
                x = x.to(self.device)
                heart_labels = heart_labels.to(self.device)
                id_labels = id_labels.to(self.device)

                # 生成噪声并添加
                noise = self.processor(x)
                x_processed = x + noise

                # 预测
                heart_outputs = self.heart_model(x_processed)
                id_outputs = self.id_model(x_processed)

                # 计算准确率
                heart_preds = torch.argmax(heart_outputs, dim=1)
                id_preds = torch.argmax(id_outputs, dim=1)
                heart_acc = (heart_preds == heart_labels).float().mean()
                id_acc = (id_preds == id_labels).float().mean()

                total_heart_acc += heart_acc.item()
                total_id_acc += id_acc.item()
                num_batches += 1

        return {
            'heart_acc': total_heart_acc / num_batches,
            'id_acc': total_id_acc / num_batches
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
        # 练阶段
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

        # 验证阶���
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
    绘制原始信号与处理后信的对比图
    Args:
        test_loader: 测试数据加载器
        processor: 隐私处理模型
        device: 设备（CPU/GPU）
        num_samples: 绘制的样本数量
    """
    processor.eval()
    raw_signals = []
    processed_signals = []

    # 提前 num_samples 个信号
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


def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def load_model(model, path, device):
    """加载模型"""
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {path}: {str(e)}")
        return None


def prepare_data_loaders(dataset, batch_size=2, train_ratio=0.8, num_workers=2):  # 将num_workers改为2
    """准备数据加载器"""
    # 计算训练集和测试集大小
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # 随机划分数据集
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True  # 添加这个参数以重用workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True  # 添加这个参数以重用workers
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Training set size: {train_size}")
    print(f"Test set size: {test_size}")
    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Test Loader batches: {len(test_loader)}")

    return train_loader, test_loader


def visualize_analysis_results(analysis_results):
    """可视化分析结果"""
    plt.style.use('seaborn-v0_8')  # 更新为新版本的样式名称

    # 创建保存目录
    os.makedirs('analysis_results/plots', exist_ok=True)

    # 1. 频段分析结果可视化
    def plot_band_analysis():
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # 获取频段名称和数据
        bands = list(analysis_results['band_discrimination'].keys())
        heart_values = [analysis_results['band_discrimination'][band]['heart'] for band in bands]
        id_values = [analysis_results['band_discrimination'][band]['id'] for band in bands]
        ratio_values = [analysis_results['band_discrimination'][band]['ratio'] for band in bands]

        # 绘制判别比对比图
        ax = axes[0]
        x = np.arange(len(bands))
        width = 0.35
        ax.bar(x - width / 2, heart_values, width, label='Heart Task')
        ax.bar(x + width / 2, id_values, width, label='ID Task')
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.set_title('Frequency Band Discrimination')
        ax.set_ylabel('Fisher Ratio')
        ax.legend()
        ax.grid(True)

        # 绘制ID/心跳比值
        ax = axes[1]
        ax.bar(bands, ratio_values)
        ax.set_title('ID/Heart Discrimination Ratio')
        ax.set_ylabel('Ratio')
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands, rotation=45)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('analysis_results/plots/band_discrimination.png', dpi=300, bbox_inches='tight')
        plt.close()

        return fig

    # 2. 频段能量分布可视化
    def plot_band_ranges():
        fig, ax = plt.subplots(figsize=(10, 6))

        bands = analysis_results['freq_bands']
        y_positions = np.arange(len(bands))

        # 绘制频段范围
        for i, (band_name, (low, high)) in enumerate(bands.items()):
            ax.plot([low, high], [i, i], 'b-', linewidth=2)
            ax.plot([low], [i], 'b|', markersize=10)
            ax.plot([high], [i], 'b|', markersize=10)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(bands.keys())
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title('Frequency Band Ranges')
        ax.grid(True)

        plt.savefig('analysis_results/plots/band_ranges.png', dpi=300, bbox_inches='tight')
        plt.close()

        return fig

    # 生成所有可视化结果
    figs = {
        'band_analysis': plot_band_analysis(),
        'band_ranges': plot_band_ranges()
    }

    # 生成总结报告
    summary = "=== 频段分析总结 ===\n\n"
    for band, data in analysis_results['band_discrimination'].items():
        summary += f"{band} 频段:\n"
        summary += f"  心跳任务判别比: {data['heart']:.4f}\n"
        summary += f"  ID任务判别比: {data['id']:.4f}\n"
        summary += f"  ID/心跳比值: {data['ratio']:.4f}\n\n"

    # 保存总结报告
    with open('analysis_results/analysis_summary.txt', 'w') as f:
        f.write(summary)

    return figs


def generate_analysis_report(analysis_results, figs):
    """成分析报告"""
    report = {
        'summary': {
            'key_findings': [],
            'recommendations': []
        },
        'detailed_analysis': {
            'frequency_domain': {},
            'time_domain': {},
            'ml_features': {}
        }
    }

    # 1. 分析频域结果
    freq_results = analysis_results['frequency_analysis']
    band_discrim = freq_results['band_discrimination']

    # ��出最��区分度的频段
    best_band = max(band_discrim.items(), key=lambda x: x[1])
    report['summary']['key_findings'].append(
        f"最具区分度的频段是 {best_band[0]}，Fisher比值为 {best_band[1]:.2f}"
    )

    # 2. 分析时域结果
    time_results = analysis_results['time_analysis']
    # 添加时域分析的关键发现

    # 3. 分析机器学习结果
    ml_results = analysis_results['ml_analysis']

    # 特征重要性分析
    heart_importance = ml_results['feature_importance']['heart']
    id_importance = ml_results['feature_importance']['id']

    # 找出对每个任务最重要的特征
    top_heart_features = np.argsort(heart_importance)[-3:]
    top_id_features = np.argsort(id_importance)[-3:]

    report['summary']['key_findings'].extend([
        f"心跳分类最重要的特征索引: {top_heart_features}",
        f"身份识别最重要的特征索引: {top_id_features}"
    ])

    # 4. 生成建议
    report['summary']['recommendations'].extend([
        "建议的频段���分方���：",
        f"  - 心跳特征保持频段: {best_band[0]}",
        "  - 身份特征扰动频段: " +
        ", ".join([b for b, v in band_discrim.items() if v < np.median(list(band_discrim.values()))])
    ])

    return report


def save_analysis_results(analysis_results, save_dir='analysis_results'):
    """保存分析结果和图像"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存频段分析图
    freq_dir = os.path.join(save_dir, 'frequency')
    os.makedirs(freq_dir, exist_ok=True)

    # 获取频段名称列表
    band_names = list(analysis_results['band_discrimination'].keys())

    # 频段判别比对比图
    plt.figure(figsize=(12, 6))
    heart_values = [analysis_results['band_discrimination'][band]['heart'] for band in band_names]
    id_values = [analysis_results['band_discrimination'][band]['id'] for band in band_names]

    x = np.arange(len(band_names))
    width = 0.35
    plt.bar(x - width / 2, heart_values, width, label='Heart Task')
    plt.bar(x + width / 2, id_values, width, label='ID Task')
    plt.xticks(x, band_names, rotation=45)
    plt.title('Frequency Band Discrimination')
    plt.ylabel('Fisher Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(freq_dir, 'band_discrimination.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 保存频段范围图
    plt.figure(figsize=(10, 6))
    y_positions = np.arange(len(band_names))

    for i, band_name in enumerate(band_names):
        low, high = analysis_results['freq_bands'][band_name]
        plt.plot([low, high], [i, i], 'b-', linewidth=2)
        plt.plot([low], [i], 'b|', markersize=10)
        plt.plot([high], [i], 'b|', markersize=10)

    plt.yticks(y_positions, band_names)
    plt.xlabel('Frequency (Hz)')
    plt.title('Frequency Band Ranges')
    plt.grid(True)
    plt.savefig(os.path.join(freq_dir, 'band_ranges.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 保存ID/心跳比值图
    plt.figure(figsize=(10, 6))
    ratio_values = [analysis_results['band_discrimination'][band]['ratio'] for band in band_names]
    x = np.arange(len(band_names))
    plt.bar(x, ratio_values)
    plt.xticks(x, band_names, rotation=45)
    plt.title('ID/Heart Discrimination Ratio')
    plt.ylabel('Ratio')
    plt.grid(True)
    plt.savefig(os.path.join(freq_dir, 'id_heart_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 保存分析报告
    report_path = os.path.join(save_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== ECG频段分析报告 ===\n\n")

        # 频段分析结果
        f.write("1. 频段分析结果:\n")
        f.write("-----------------\n")
        for band_name in band_names:
            data = analysis_results['band_discrimination'][band_name]
            f.write(
                f"\n{band_name}频段 ({analysis_results['freq_bands'][band_name][0]}-{analysis_results['freq_bands'][band_name][1]} Hz):\n")
            f.write(f"  - 心跳任务判别比: {data['heart']:.4f}\n")
            f.write(f"  - ID任务判别比: {data['id']:.4f}\n")
            f.write(f"  - ID/心跳比值: {data['ratio']:.4f}\n")

        # 关键发现
        f.write("\n2. 关键发现:\n")
        f.write("------------\n")

        # 找出心跳任务最重要的频段
        heart_important = max(analysis_results['band_discrimination'].items(),
                              key=lambda x: x[1]['heart'])
        f.write(f"- 心跳任务最重要的频段: {heart_important[0]} "
                f"(判别比: {heart_important[1]['heart']:.4f})\n")

        # 找出ID任务最重要的频段
        id_important = max(analysis_results['band_discrimination'].items(),
                           key=lambda x: x[1]['id'])
        f.write(f"- ID任务最重要的频段: {id_important[0]} "
                f"(判别比: {id_important[1]['id']:.4f})\n")

        # 找出ID/心跳比值最高的频段
        ratio_important = max(analysis_results['band_discrimination'].items(),
                              key=lambda x: x[1]['ratio'])
        f.write(f"- ID/心跳比值最高的频段: {ratio_important[0]} "
                f"(比值: {ratio_important[1]['ratio']:.4f})\n")

        # 建议
        f.write("\n3. 隐私保护建议:\n")
        f.write("---------------\n")
        f.write("基于分析结果，建议:\n")
        f.write(f"1. 保留 {heart_important[0]} 频段以保持心跳信息\n")
        f.write(f"2. 对 {id_important[0]} 频段进行重点处理以保护隐私\n")
        f.write(f"3. 特别关注 {ratio_important[0]} 频段，"
                "该频段对ID识别的贡献显著高于心跳识别\n")

    print(f"分析结果已保存到 {save_dir} 目录")


def analyze_ecg_data(train_loader, test_loader, device):
    """分析ECG数据特征，进行精细频段分析"""
    print("开始分析ECG数据...")

    # 创建分析器实例
    analyzer = ECGAnalyzer(sampling_rate=360)

    # 初始化累积变量
    total_samples = 0
    accumulated_psd = None
    heart_label_counts = None
    id_label_counts = None

    print("第一遍扫描：计算频谱...")
    total_batches = len(train_loader)

    with torch.no_grad():
        for batch_idx, (signals, heart_labels, id_labels) in enumerate(train_loader):
            # 处理当前批次数据
            signals = signals.squeeze(1).cpu().numpy()
            batch_size = signals.shape[0]

            # 计算当前批次的PSD
            freqs, batch_psd = signal.welch(signals, fs=360,
                                            nperseg=min(signals.shape[1], 1024))

            # 累积PSD
            if accumulated_psd is None:
                accumulated_psd = np.zeros_like(batch_psd[0])
            accumulated_psd += np.sum(batch_psd, axis=0)

            # 累积样本数
            total_samples += batch_size

            # 显示进度
            if (batch_idx + 1) % 10 == 0:
                print(f"\r处理进度: {batch_idx + 1}/{total_batches} "
                      f"({(batch_idx + 1) / total_batches * 100:.1f}%)", end="")

    # 计算平均PSD
    average_psd = accumulated_psd / total_samples

    # 创建360个频段
    print("\n\n进行精细频段分析...")
    max_freq = 180  # 奈奎斯特频率
    freq_step = max_freq / 360  # 每个频段的宽度

    band_discrimination = {}
    freq_bands = {}

    # 预先计算所有频段的mask
    band_masks = {}
    for i in range(360):
        low_freq = i * freq_step
        high_freq = (i + 1) * freq_step
        band_name = f"band_{i:03d}"
        freq_bands[band_name] = (low_freq, high_freq)
        band_masks[band_name] = (freqs >= low_freq) & (freqs < high_freq)

    # 初始化每个频段的累积数据
    band_energies = {band: {'heart': {}, 'id': {}} for band in freq_bands.keys()}

    print("第二遍扫描：计算频段能量...")
    for batch_idx, (signals, heart_labels, id_labels) in enumerate(train_loader):
        signals = signals.squeeze(1).cpu().numpy()
        heart_labels = heart_labels.cpu().numpy()
        id_labels = id_labels.squeeze().cpu().numpy()

        # 计算当前批次的PSD
        _, batch_psd = signal.welch(signals, fs=360,
                                    nperseg=min(signals.shape[1], 1024))

        # 对每个频段计算能量
        for band_name, mask in band_masks.items():
            band_energy = np.sum(batch_psd[:, mask], axis=1)

            # 累积心跳标签的能量
            for j in range(heart_labels.shape[1]):
                mask = heart_labels[:, j] == 1
                if np.any(mask):
                    if j not in band_energies[band_name]['heart']:
                        band_energies[band_name]['heart'][j] = []
                    band_energies[band_name]['heart'][j].extend(band_energy[mask])

            # 累积ID标签的能量
            for id_ in np.unique(id_labels):
                mask = id_labels == id_
                if np.any(mask):
                    if id_ not in band_energies[band_name]['id']:
                        band_energies[band_name]['id'][id_] = []
                    band_energies[band_name]['id'][id_].extend(band_energy[mask])

        # 显示进度
        if (batch_idx + 1) % 10 == 0:
            print(f"\r处理进度: {batch_idx + 1}/{total_batches} "
                  f"({(batch_idx + 1) / total_batches * 100:.1f}%)", end="")

    print("\n\n计算判别比...")
    # 计算每个频段的判别比
    for band_name in freq_bands.keys():
        # 转换累积的列表为numpy数组
        heart_energies = {k: np.array(v) for k, v in band_energies[band_name]['heart'].items()}
        id_energies = {k: np.array(v) for k, v in band_energies[band_name]['id'].items()}

        # 计算判别比
        heart_discrim = analyzer._calculate_discrimination(heart_energies) if heart_energies else 0.0
        id_discrim = analyzer._calculate_discrimination(id_energies) if id_energies else 0.0

        band_discrimination[band_name] = {
            'heart': heart_discrim,
            'id': id_discrim,
            'ratio': id_discrim / (heart_discrim + 1e-10),
            'freq_points': np.sum(band_masks[band_name])
        }

    print("\n分析完成，生成总结...")

    # 找出重要的频段
    heart_important = sorted(band_discrimination.items(),
                             key=lambda x: x[1]['heart'],
                             reverse=True)[:10]
    id_important = sorted(band_discrimination.items(),
                          key=lambda x: x[1]['id'],
                          reverse=True)[:10]
    ratio_important = sorted(band_discrimination.items(),
                             key=lambda x: x[1]['ratio'],
                             reverse=True)[:10]

    # 打印结果
    print("\n心跳任务最重要的10个频段:")
    for band, data in heart_important:
        freq_range = freq_bands[band]
        print(f"{band} ({freq_range[0]:.2f}-{freq_range[1]:.2f} Hz): {data['heart']:.4f}")

    print("\nID任务最重要的10个频段:")
    for band, data in id_important:
        freq_range = freq_bands[band]
        print(f"{band} ({freq_range[0]:.2f}-{freq_range[1]:.2f} Hz): {data['id']:.4f}")

    print("\nID/心跳比值最高的10个频段:")
    for band, data in ratio_important:
        freq_range = freq_bands[band]
        print(f"{band} ({freq_range[0]:.2f}-{freq_range[1]:.2f} Hz): {data['ratio']:.4f}")

    # 保存结果
    results = {
        'band_discrimination': band_discrimination,
        'freq_bands': freq_bands,
        'sampling_rate': 360,
        'important_bands': {
            'heart': heart_important,
            'id': id_important,
            'ratio': ratio_important
        }
    }

    # 保存详细分析结果
    os.makedirs('analysis_results', exist_ok=True)
    np.save('analysis_results/detailed_band_discrimination.npy', band_discrimination)

    return results


def main():
    # 设置模型保存路径
    MODEL_DIR = './saved_models'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    HEART_MODEL_PATH = os.path.join(MODEL_DIR, 'heart_model.pth')
    ID_MODEL_PATH = os.path.join(MODEL_DIR, 'id_model.pth')

    # ��置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. 准备数据
    dataset = ECGDataset('../data/raw/mit/')  # 自动下载和处理数据
    train_loader, test_loader = prepare_data_loaders(dataset)

    # 2. 初始化���型
    heart_model = ECGClassifier(num_classes=5, is_multilabel=True)
    id_model = ECGClassifier(num_classes=len(dataset.id_to_idx), is_multilabel=False)

    # 3. 检查是否存在已保存的模型
    models_exist = os.path.exists(HEART_MODEL_PATH) and os.path.exists(ID_MODEL_PATH)

    if models_exist:
        logger.info("找到保存模型，正在加载...")
        # 加载已保存的模型
        heart_model = load_model(heart_model, HEART_MODEL_PATH, device)
        id_model = load_model(id_model, ID_MODEL_PATH, device)

        if heart_model is None or id_model is None:
            logger.error("模型加载失败，将新训练")
            models_exist = False

    if not models_exist:
        logger.info("未找到已保存的模型，开始训练...")
        # 训练心跳分类模型
        logger.info("Training heart beat classifier...")
        heart_model = train_classifier(
            heart_model, train_loader, test_loader, device, is_multilabel=True)
        # 保存心跳模型
        save_model(heart_model, HEART_MODEL_PATH)

        # 训练ID识别模型
        logger.info("Training ID classifier...")
        id_model = train_classifier(
            id_model, train_loader, test_loader, device, is_multilabel=False)
        # 保存ID型
        save_model(id_model, ID_MODEL_PATH)

    # 将模型移到设备上并设置为评估模式
    heart_model = heart_model.to(device)
    id_model = id_model.to(device)
    heart_model.eval()
    id_model.eval()

    # 好的 那我们现在这里  先只用很多方法去分析数据  先不用做数据处理去除隐私
    # 因为当我对不同频段有清晰的分析  我们就可以精准的不大需要改动数据 达到我们想要的隐私保护的数据效果

    # 分析数据
    analysis_results = analyze_ecg_data(train_loader, test_loader, device)

    # 查看分析结果
    visualize_analysis_results(analysis_results)

    # 保存分析结果和图像
    save_analysis_results(analysis_results)


if __name__ == '__main__':
    main()