# import wfdb
#
# # 下载MIT-BIH数据集  也有很多别的数据集 后面可以换
# wfdb.dl_database('mitdb', './data/raw/mit/')
#
import torch.nn.functional as F

# 然后读取
import wfdb
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class EcgBeatDataset(Dataset):
    def __init__(self, beat_types, qrs_range, beat_length, data_dir):
        self.beat_types = beat_types
        self.qrs_range = qrs_range
        self.beat_length = beat_length
        self.data_dir = data_dir
        self.data = []
        self.heart_labels = []
        self.id_labels = []

        # 定义患者 ID 映射
        id_mapping = {
            '201': '201',  # 201 和 202 是同一患者
            '202': '201'
        }

        # 获取所有记录列表
        record_list = wfdb.get_record_list('mitdb')

        # 为未映射的记录赋默认值
        for record in record_list:
            if record not in id_mapping:
                id_mapping[record] = record  # 默认使用自己的编号

        # 将患者 ID 映射为连续整数
        unique_ids = sorted(set(id_mapping.values()))
        patient_id_map = {patient: idx for idx, patient in enumerate(unique_ids)}

        # 遍历记录列表
        for record_name in record_list:
            if record_name not in id_mapping:
                print(f"Skipping record {record_name}: not mapped to any patient.")
                continue

            record_path = os.path.join(self.data_dir, record_name)
            patient_label = patient_id_map[id_mapping[record_name]]  # 获取整数患者标签

            # 检查文件是否存在
            if not os.path.exists(f"{record_path}.dat"):
                print(f"Skipping missing record: {record_name}")
                continue

            # 读取记录和注释
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            # 遍历目标心跳类型
            for beat_type_idx, beat_type in enumerate(beat_types):
                qrs_peaks = [index for index, value in enumerate(annotation.symbol) if value == beat_type]

                # 提取心跳片段
                for qrs_peak in qrs_peaks:
                    # 确定信号的开始和结束位置
                    start = annotation.sample[qrs_peak] - (beat_length // 2)
                    end = annotation.sample[qrs_peak] + (beat_length // 2)

                    # 处理边界情况
                    if start < 0:  # 前面不足
                        ecg_beat = record.p_signal[0:end, 0]
                        padding = beat_length - len(ecg_beat)
                        ecg_beat = np.pad(ecg_beat, (padding, 0), mode='edge')  # 用第一个值填充前端
                    elif end > len(record.p_signal):  # 后面不足
                        ecg_beat = record.p_signal[start:, 0]
                        padding = beat_length - len(ecg_beat)
                        ecg_beat = np.pad(ecg_beat, (0, padding), mode='edge')  # 用最后一个值填充后端
                    else:  # 正常片段
                        ecg_beat = record.p_signal[start:end, 0]

                    # 信号归一化
                    if np.std(ecg_beat) != 0:
                        ecg_beat = (ecg_beat - np.mean(ecg_beat)) / np.std(ecg_beat)
                    else:
                        ecg_beat = ecg_beat - np.mean(ecg_beat)

                    # 保存结果
                    self.data.append(torch.tensor(ecg_beat).float())
                    self.heart_labels.append(beat_type_idx)
                    self.id_labels.append(patient_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].unsqueeze(0)  # 增加通道维度
        heart_label = torch.tensor(self.heart_labels[idx], dtype=torch.long)
        id_label = torch.tensor(self.id_labels[idx], dtype=torch.long)
        return sample, heart_label, id_label

data_dir = '../data/raw/mit/'
qrs_range = 60
beat_length = 360
beat_types = ['N', 'A', 'V', 'F', 'Q']

# 创建数据集实例
dataset = EcgBeatDataset(beat_types, qrs_range, beat_length, data_dir)


# 从数据集中提取样本
sample, heart_label, id_label = dataset[0]

print(f"Sample shape: {sample.shape}")
print(f"Heart label: {heart_label}")
print(f"Patient ID label: {id_label}")

# 统计总样本数量
total_samples = len(dataset)
print(f"Total number of samples: {total_samples}")

# 统计心跳类型标签的数量
heart_labels = torch.tensor(dataset.heart_labels)
heart_label_counts = torch.bincount(heart_labels)
print(f"Heart label counts: {heart_label_counts}")

# 统计病人 ID 标签的数量
id_labels = torch.tensor(dataset.id_labels)
id_label_counts = torch.bincount(id_labels)
print(f"ID label counts: {id_label_counts}")

# 打印每个样本的维度
sample_shape = dataset[0][0].shape
print(f"Sample shape: {sample_shape}")
import torch
import torch.nn as nn



class PatientIDCNN(nn.Module):
    def __init__(self, num_classes=48):
        super(PatientIDCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(0.5)

        self.id_conv = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.id_bn = nn.BatchNorm1d(256)
        self.id_pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(5632, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        x = self.id_pool(torch.relu(self.id_bn(self.id_conv(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HeartbeatCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(HeartbeatCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(0.5)

        self.heart_conv = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.heart_bn = nn.BatchNorm1d(256)
        self.heart_pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(5632, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        x = self.heart_pool(torch.relu(self.heart_bn(self.heart_conv(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import torch.optim as optim


def train_heart_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    best_val_loss = float('inf')
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        best_acc = 0.0

        for inputs, heart_labels, _ in train_loader:  # 只取心跳分类标签
            inputs, heart_labels = inputs.to(device).float(), heart_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, heart_labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == heart_labels).sum().item()
            total += heart_labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        accuracy = correct / total

        # 验证模型性能
        val_loss, val_acc = evaluate_heart_model(model, val_loader, criterion, device)

        # 保存最佳模型权重
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 保存最佳模型到文件
    torch.save(model.state_dict(), 'model_mit_heart.pth')
    print("最佳心跳分类模型已保存为 model_mit_heart.pth")


def train_id_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    best_val_loss = float('inf')
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        best_acc = 0

        for inputs, _, id_labels in train_loader:  # 只取ID分类标签
            inputs, id_labels = inputs.to(device).float(), id_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, id_labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == id_labels).sum().item()
            total += id_labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        accuracy = correct / total

        # 验证模型性能
        val_loss, val_acc = evaluate_id_model(model, val_loader, criterion, device)

        # 保存最佳模型权重
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 保存最佳模型到文件
    torch.save(model.state_dict(), 'model_mit_id.pth')
    print("最佳病人ID分类模型已保存为 model_mit_id.pth")


def evaluate_heart_model(model, loader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, heart_labels, _ in loader:
            inputs, heart_labels = inputs.to(device), heart_labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute validation loss
            loss = criterion(outputs, heart_labels)
            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == heart_labels).sum().item()
            total += heart_labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_id_model(model, loader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, _, id_labels in loader:
            inputs, id_labels = inputs.to(device), id_labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute validation loss
            loss = criterion(outputs, id_labels)
            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == id_labels).sum().item()
            total += id_labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

# 将数据集划分为训练集、验证集和测试集
train_size = 0.7
val_size = 0.15
test_size = 0.15

# 获取数据集的索引列表
dataset_size = len(dataset)
indices = list(range(dataset_size))

# 使用train_test_split划分训练集、验证集和测试集
train_indices, temp_indices = train_test_split(indices, train_size=train_size, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, train_size=val_size / (val_size + test_size),
                                             random_state=42)

# 创建Subset和DataLoader
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_label_counts(loader, label_type="heart"):
    """
    打印DataLoader中每种标签的数量。
    Args:
        loader: DataLoader对象。
        label_type: 要统计的标签类型 ('heart' 或 'id')。
    """
    labels = []

    for _, heart_labels, id_labels in loader:
        if label_type == "heart":
            labels.extend(heart_labels.cpu().numpy())
        elif label_type == "id":
            labels.extend(id_labels.cpu().numpy())
        else:
            raise ValueError("Invalid label_type. Use 'heart' or 'id'.")

    # 统计标签数量
    labels = np.array(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"Counts for {label_type} labels:")
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} samples")
    print(f"Total {label_type} labels: {len(labels)}\n")


# 打印每种ID标签和心跳标签的数量
print("Test Loader Statistics:")
print_label_counts(test_loader, label_type="heart")
print_label_counts(test_loader, label_type="id")


# 打印每种ID标签和心跳标签的数量
print("train Loader Statistics:")
print_label_counts(train_loader, label_type="heart")
print_label_counts(train_loader, label_type="id")


# 打印每种ID标签和心跳标签的数量
print("val Loader Statistics:")
print_label_counts(val_loader, label_type="heart")
print_label_counts(val_loader, label_type="id")


# 定义损失函数和优化器
heart_criterion = nn.CrossEntropyLoss()
id_criterion = nn.CrossEntropyLoss()


# Before training, print the unique labels to check their range
print(f"Unique heart labels: {torch.unique(torch.tensor([label for _, label, _ in dataset]))}")
print(f"Unique id labels: {torch.unique(torch.tensor([id_label for _, _, id_label in dataset]))}")


heart_model = HeartbeatCNN(num_classes=5).to(device)
heart_optimizer = optim.Adam(heart_model.parameters(), lr=0.001)
# train_heart_model(heart_model, train_loader, val_loader, heart_criterion, heart_optimizer, num_epochs=20, device=device)

# 实例化病人ID分类模型并训练
id_model = PatientIDCNN(num_classes=47).to(device)
id_optimizer = optim.Adam(id_model.parameters(), lr=0.001)
# train_id_model(id_model, train_loader, val_loader, id_criterion, id_optimizer, num_epochs=20, device=device)


# 加载心跳分类模型
model_heart = HeartbeatCNN(num_classes=5).to(device)
model_heart.load_state_dict(torch.load('model_mit_heart.pth'))

# 加载病人ID分类模型
model_id = PatientIDCNN(num_classes=47).to(device)
model_id.load_state_dict(torch.load('model_mit_id.pth'))

# 在测试集上评估心跳分类模型
test_heart_loss, test_heart_acc = evaluate_heart_model(model_heart, test_loader, heart_criterion, device=device)
print(f'Test Heart Loss: {test_heart_loss:.4f}, Test Heart Accuracy: {test_heart_acc:.4f}')

# 在测试集上评估病人ID分类模型
test_id_loss, test_id_acc = evaluate_id_model(model_id, test_loader, id_criterion, device=device)
print(f'Test ID Loss: {test_id_loss:.4f}, Test ID Accuracy: {test_id_acc:.4f}')




import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft
import matplotlib.pyplot as plt

def plot_ecg_comparison(loader1, loader2, num_samples=50, save_path="ecg_comparison.png"):
    # 随机选择50个样本的索引
    # indices = random.sample(range(len(loader1.dataset)), num_samples)
    indices = list(range(num_samples))
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))

    for i, idx in enumerate(indices):
        # 从两个数据加载器中获取相应的样本
        (reconstructed, _, _), (original, _, _) = loader1.dataset[idx], loader2.dataset[idx]

        # 确保张量在 CPU 上并转换为 numpy 数组
        reconstructed_signal = reconstructed.squeeze().cpu().numpy()
        original_signal = original.squeeze().cpu().numpy()

        # 绘制原始信号
        axes[i, 0].plot(original_signal)
        axes[i, 0].set_title(f"Original Signal {idx}")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].set_xlabel("Sample")

        # 绘制带噪声的信号
        axes[i, 1].plot(reconstructed_signal)
        axes[i, 1].set_title(f"Noisy Signal {idx}")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].set_xlabel("Sample")

    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # 关闭图像以释放内存
    print(f"ECG comparison plot saved to {save_path}")



def compute_frequency_importance(model, loader, fs=360, freq_bins=10, device='cpu'):
    """
    计算频率域重要性，基于模型对不同频段特征的敏感性。
    """
    model.eval()
    total_importance = None

    # 定义频段边界
    freq_edges = np.linspace(0, fs // 2, freq_bins + 1)  # 按照freq_bins分割频率范围

    for data, labels, _ in loader:
        data = data.to(device)
        data.requires_grad = True

        # FFT 分解
        data_freq = torch.fft.rfft(data, dim=-1)
        freq_axis = torch.fft.rfftfreq(data.size(-1), d=1 / fs).to(device)

        # 预测并计算梯度
        outputs = model(data)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels.to(device))
        loss.backward()

        # 计算频率梯度
        grad = data.grad
        grad_freq = torch.fft.rfft(grad, dim=-1).abs().mean(dim=0)

        # 初始化或累加
        if total_importance is None:
            total_importance = torch.zeros(freq_bins, device=device)

        # 将频率梯度按频段分类并累加
        for i in range(freq_bins):
            band_mask = (freq_axis >= freq_edges[i]) & (freq_axis < freq_edges[i + 1])
            total_importance[i] += grad_freq[:, band_mask].sum().item()

    # 归一化
    total_importance /= total_importance.sum()
    return total_importance.cpu().numpy(), freq_edges


def plot_frequency_heatmap(freq_importance_id, freq_importance_heart, freq_edges, save_path="frequency_heatmap.png"):
    """
    绘制频率域特征重要性的热力图。
    """
    # 调整边界和标签数量匹配
    if len(freq_edges) == len(freq_importance_id) + 1:
        freq_labels = [f"{freq_edges[i]:.1f}-{freq_edges[i+1]:.1f} Hz" for i in range(len(freq_edges) - 1)]
    else:
        raise ValueError("Frequency edges do not match frequency importance length.")

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


def compute_freq_guidance_mask(freq_importance_id, freq_importance_heart, device):
    """
    根据频率重要性计算指导掩模。
    """
    if len(freq_importance_id) != len(freq_importance_heart):
        raise ValueError("ID and Heart frequency importance lengths do not match.")

    freq_guidance = freq_importance_id - freq_importance_heart
    # freq_guidance[freq_guidance < 0] = 0  # 保留对 ID 敏感的频段
    # freq_guidance = (freq_importance_id - freq_importance_heart) ** 2
    freq_guidance[freq_guidance < 0] = 0

    # 转换为张量
    return torch.tensor(freq_guidance, dtype=torch.float32).to(device)

def compute_frequency_distribution(loader, fs=360, freq_bins=10, device='cpu'):
    """
    统计 loader 中所有样本在不同频段的分布。
    Args:
        loader: 数据加载器，包含时域信号。
        fs: 采样率（Hz）。
        freq_bins: 频段数量。
        device: 运行设备（'cpu' 或 'cuda'）。
    Returns:
        freq_distribution: 每个频段的总幅值分布。
        freq_edges: 每个频段的频率边界。
    """
    # 定义频段边界
    freq_edges = np.linspace(0, fs // 2, freq_bins + 1)  # [0, fs/2] 等分为 freq_bins 段
    freq_distribution = np.zeros(freq_bins)  # 初始化分布

    for data, _, _ in loader:
        data = data.to(device)

        # 对每个样本进行傅里叶变换
        data_freq = torch.fft.rfft(data, dim=-1)  # RFFT: 只计算正频率部分
        data_magnitude = torch.abs(data_freq).mean(dim=1).cpu().numpy()  # 每个样本的频率幅值
        freq_axis = torch.fft.rfftfreq(data.size(-1), d=1 / fs).cpu().numpy()  # 频率轴

        # 将幅值累加到对应频段
        for i in range(freq_bins):
            band_mask = (freq_axis >= freq_edges[i]) & (freq_axis < freq_edges[i + 1])
            freq_distribution[i] += data_magnitude[:, band_mask].sum()  # 对每个频段累加幅值

    # 归一化分布
    freq_distribution /= freq_distribution.sum()

    return freq_distribution, freq_edges


def plot_frequency_distribution(freq_distribution, freq_edges, loader_name="loader", save_path="freq_distribution.png"):
    """
    绘制频段分布的柱状图。
    Args:
        freq_distribution: 每个频段的归一化幅值分布。
        freq_edges: 每个频段的频率边界。
        loader_name: 数据加载器的名称，用于图的标题。
        save_path: 图像保存路径。
    """
    # 生成频段标签
    freq_labels = [f"{freq_edges[i]:.1f}-{freq_edges[i+1]:.1f} Hz" for i in range(len(freq_distribution))]

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(freq_distribution)), freq_distribution, width=0.8, tick_label=freq_labels)
    plt.title(f"Frequency Distribution for {loader_name}")
    plt.xlabel("Frequency Range (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path)
    plt.show()
    print(f"Frequency distribution plot saved to {save_path}")


def create_noisy_test_loader(unet, test_loader, device):
    """
    使用训练好的 UNet 生成带噪的测试数据加载器。
    """
    unet.eval()
    noisy_data = []
    heart_labels = []
    id_labels = []

    with torch.no_grad():
        for data, heart_label, id_label in test_loader:
            data = data.to(device)
            noise = unet(data)  # 生成噪声
            perturbed_data = data + noise  # 添加噪声

            noisy_data.append(perturbed_data.cpu())
            heart_labels.append(heart_label)
            id_labels.append(id_label)

    # 创建新的 DataLoader
    noisy_dataset = torch.utils.data.TensorDataset(torch.cat(noisy_data), torch.cat(heart_labels), torch.cat(id_labels))
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=test_loader.batch_size, shuffle=False)
    return noisy_loader


freq_importance_id, freq_edges = compute_frequency_importance(
    model_id, train_loader, fs=360, freq_bins=360, device=device
)
freq_importance_heart, _ = compute_frequency_importance(
    model_heart, train_loader, fs=360, freq_bins=360, device=device
)

print(f"Length of freq_edges: {len(freq_edges)}")
print(f"Length of freq_importance_id: {len(freq_importance_id)}")
print(f"Length of freq_importance_heart: {len(freq_importance_heart)}")

plot_frequency_heatmap(freq_importance_id, freq_importance_heart, freq_edges, save_path="train_loaderfrequency_heatmap_importance.png")

freq_guidance = compute_freq_guidance_mask(freq_importance_id, freq_importance_heart, device=device)

print(freq_guidance)


freq_distribution, freq_edges = compute_frequency_distribution(train_loader, fs=360, freq_bins=36, device='cuda')

plot_frequency_distribution(freq_distribution, freq_edges, loader_name="train_loader", save_path="freq_distribution_train_loader.png")

from scipy.signal import butter, filtfilt
class ImprovedUNetWithFrequencyMask(nn.Module):
    def __init__(self, input_dim, freq_guidance, fs=360, noise_limit=0.005):
        super(ImprovedUNetWithFrequencyMask, self).__init__()
        self.freq_guidance = freq_guidance
        self.fs = fs
        self.input_dim = input_dim
        self.noise_limit = noise_limit  # 噪声幅度限制

        # 编码器部分
        self.encoder1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # 中间层
        self.middle = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # 解码器部分
        self.decoder3 = nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1)
        self.decoder2 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.decoder1 = nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1)
        self.final = nn.ConvTranspose1d(16, 1, kernel_size=3, padding=1)

        # 设计 Butterworth 低通滤波器
        nyquist = 0.5 * 360
        normalized_cutoff = 60 / nyquist
        self.b, self.a = butter(4, normalized_cutoff, btype='low', analog=False)

    def forward(self, x):
        # 编码阶段
        e1 = F.relu(self.encoder1(x))
        e2 = F.relu(self.encoder2(F.max_pool1d(e1, kernel_size=2)))
        e3 = F.relu(self.encoder3(F.max_pool1d(e2, kernel_size=2)))

        # 中间层
        m = F.relu(self.middle(F.max_pool1d(e3, kernel_size=2)))

        # 解码阶段
        d3 = F.relu(self.decoder3(F.interpolate(m, scale_factor=2, mode='linear', align_corners=True)) + e3)
        d2 = F.relu(self.decoder2(F.interpolate(d3, scale_factor=2, mode='linear', align_corners=True)) + e2)
        d1 = F.relu(self.decoder1(F.interpolate(d2, scale_factor=2, mode='linear', align_corners=True)) + e1)
        decoded = self.final(d1)

        noise = decoded
        # 将噪声从 PyTorch 张量转为 NumPy 数组进行滤波
        noise_np = noise.detach().cpu().numpy()

        # 应用低通滤波器，仅保留 50 Hz 以下频率分量
        filtered_noise_np = []
        for batch_noise in noise_np:
            filtered_signal = filtfilt(self.b, self.a, batch_noise)  # 滤波
            filtered_noise_np.append(filtered_signal)

        # 转回 PyTorch 张量
        noise = torch.tensor(filtered_noise_np, dtype=torch.float32, device=x.device, requires_grad=True)

        # # 动态频率掩模
        # noise_freq = torch.fft.rfft(decoded, dim=-1)
        # freq_guidance_resized = F.interpolate(
        #     self.freq_guidance.unsqueeze(0).unsqueeze(0),
        #     size=noise_freq.size(-1),
        #     mode='linear',
        #     align_corners=True
        # ).squeeze(0).squeeze(0)
        # masked_freq = noise_freq * freq_guidance_resized
        # noise = torch.fft.irfft(masked_freq, n=self.input_dim, dim=-1)

        # 噪声裁剪：限制噪声幅度
        noise_norm = torch.norm(noise, p=2, dim=-1, keepdim=True)  # L2 范数
        scaling_factor = torch.clamp(noise_norm / self.noise_limit, min=1.0)
        noise = noise / scaling_factor

        # # 如果想检查整个张量的范围
        # print("Noise range: [{}, {}]".format(noise.min().item(), noise.max().item()))

        # 限制噪声范围
        # noise = torch.clamp(noise, min=-self.noise_limit, max=self.noise_limit)

        return noise
# unet = ImprovedUNetWithFrequencyMask(input_dim=360, freq_guidance=freq_guidance, fs=360).to(device)
unet = ImprovedUNetWithFrequencyMask(input_dim=360, freq_guidance=freq_guidance, fs=360,noise_limit=8).to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        # 动态计算 Flatten 维度
        dummy_input = torch.zeros(1, 1, input_dim)  # 假设 batch_size=1，通道数=1，长度=input_dim
        with torch.no_grad():
            self.flatten_dim = self._compute_flatten_dim(dummy_input)

        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def _compute_flatten_dim(self, x):
        """
        计算经过卷积和池化后 Flatten 的特征维度
        """
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(F.avg_pool1d(x, kernel_size=2))))
        x = F.leaky_relu(self.bn3(self.conv3(F.avg_pool1d(x, kernel_size=2))))
        x = F.avg_pool1d(x, kernel_size=2)  # 第三次下采样
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(F.avg_pool1d(x, kernel_size=2))))
        x = F.leaky_relu(self.bn3(self.conv3(F.avg_pool1d(x, kernel_size=2))))
        x = F.avg_pool1d(x, kernel_size=2)  # 第三次下采样
        x = x.view(x.size(0), -1)  # Flatten
        x = F.leaky_relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 判别器实例化
input_dim = 360  # 假设 ECG 信号长度为 360
discriminator = Discriminator(input_dim=input_dim).to(device)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)



# def train_improved_unet_with_constraints(unet, train_loader, heart_model, id_model, optimizer, num_epochs=20, device='cpu'):
def train_gan_unet_with_discriminator(unet, discriminator, train_loader, heart_model, id_model,optimizer, disc_optimizer, num_epochs=20, device='cpu'):
    """
    训练改进的 UNet，以生成有针对性的噪声，带有裁剪限制和相似性约束。
    """
    heart_model.eval()
    id_model.eval()

    # 确保 heart_model 和 id_model 的参数被冻结
    for param in heart_model.parameters():
        param.requires_grad = False
    for param in id_model.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        unet.train()
        discriminator.train()

        epoch_loss = 0.0
        epoch_heart_loss = 0.0
        epoch_id_loss = 0.0
        epoch_similarity_loss = 0.0
        epoch_noise_regularization = 0.0
        epoch_id_to_heart_loss = 0.0
        epoch_disc_loss = 0.0

        for inputs, heart_labels, id_labels in train_loader:
            inputs = inputs.to(device)
            inputs.requires_grad = True  # 确保启用梯度计算
            heart_labels = heart_labels.to(device)
            id_labels = id_labels.to(device)

            # ----------------------------
            # 1. 更新判别器
            # ----------------------------
            disc_optimizer.zero_grad()

            # 真实信号判别
            real_labels = torch.ones(inputs.size(0), 1).to(device)
            fake_labels = torch.zeros(inputs.size(0), 1).to(device)
            real_preds = discriminator(inputs)
            real_loss = F.binary_cross_entropy(real_preds, real_labels)

            # 生成噪声信号判别
            noise = unet(inputs)
            fake_signal = inputs + noise
            fake_preds = discriminator(fake_signal.detach())  # 停止生成器梯度
            fake_loss = F.binary_cross_entropy(fake_preds, fake_labels)

            # 判别器总损失
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            disc_optimizer.step()

            # epoch_disc_loss += disc_loss.item()


            # ----------------------------
            # 2. 更新生成器
            # ----------------------------

            optimizer.zero_grad()

            # 生成噪声并加到输入信号上
            noise = unet(inputs)
            perturbed_inputs = inputs + noise

            # 计算损失
            heart_outputs = heart_model(perturbed_inputs)
            id_outputs = id_model(perturbed_inputs)

            heart_loss = nn.CrossEntropyLoss()(heart_outputs, heart_labels)
            id_loss = -nn.CrossEntropyLoss()(id_outputs, id_labels)  # 对抗性损失
            id_to_heart_loss = 0.0001*nn.CrossEntropyLoss()(id_outputs, heart_labels)  # 对抗性损失
            similarity_loss = F.mse_loss(perturbed_inputs, inputs)  # 相似性约束
            noise_regularization = torch.mean(noise ** 2)  # 噪声正则化

            # 总损失
            total_loss = (
                heart_loss + 10000 * id_loss + 0.1 * noise_regularization + 0.5 * similarity_loss + id_to_heart_loss
            )
            total_loss.backward()
            optimizer.step()

            # 累加各个损失值
            epoch_loss += total_loss.item()
            epoch_heart_loss += heart_loss.item()
            epoch_id_loss += id_loss.item()
            epoch_similarity_loss += similarity_loss.item()
            epoch_noise_regularization += noise_regularization.item()
            epoch_id_to_heart_loss += id_to_heart_loss.item()
            epoch_disc_loss += disc_loss.item()

        # 打印每个损失项
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Total Loss: {epoch_loss:.4f}")
        print(f"  Heart Loss: {epoch_heart_loss:.4f}")
        print(f"  ID Loss: {epoch_id_loss:.4f}")
        print(f"  Similarity Loss: {epoch_similarity_loss:.4f}")
        print(f"  Noise Regularization: {epoch_noise_regularization:.4f}")
        print(f"  epoch_disc_loss: {epoch_disc_loss:.4f}")



# train_improved_unet_with_constraints(unet, train_loader, model_heart, model_id, optimizer, num_epochs=10, device=device)
train_gan_unet_with_discriminator(unet, discriminator, train_loader, model_heart, model_id,optimizer, disc_optimizer, num_epochs=10, device=device)
no_id_test_loader = create_noisy_test_loader(unet, test_loader, device)


# 检查加载的数据
for inputs, heart_labels, id_labels in no_id_test_loader:
    print(f"Loaded noisy inputs shape: {inputs.shape}")
    print(f"Heart labels shape: {heart_labels.shape}")
    print(f"ID labels shape: {id_labels.shape}")
    break

# 加载心跳分类模型
model_heart = HeartbeatCNN(num_classes=5).to(device)
model_heart.load_state_dict(torch.load('model_mit_heart.pth'))

# 加载病人ID分类模型
model_id = PatientIDCNN(num_classes=47).to(device)
model_id.load_state_dict(torch.load('model_mit_id.pth'))

# 在测试集上评估心跳分类模型
test_heart_loss, test_heart_acc = evaluate_heart_model(model_heart, test_loader, heart_criterion, device=device)
print(f'Test Heart Loss: {test_heart_loss:.4f}, Test Heart Accuracy: {test_heart_acc:.4f}')

# 在测试集上评估病人ID分类模型
test_id_loss, test_id_acc = evaluate_id_model(model_id, test_loader, id_criterion, device=device)
print(f'Test ID Loss: {test_id_loss:.4f}, Test ID Accuracy: {test_id_acc:.4f}')


# 在测试集上评估心跳分类模型
test_heart_loss, test_heart_acc = evaluate_heart_model(model_heart, no_id_test_loader, heart_criterion, device=device)
print(f'Test Heart Loss: {test_heart_loss:.4f}, no_id_test_loader Test Heart Accuracy: {test_heart_acc:.4f}')

# 在测试集上评估病人ID分类模型
test_id_loss, test_id_acc = evaluate_id_model(model_id, no_id_test_loader, id_criterion, device=device)
print(f'Test ID Loss: {test_id_loss:.4f},no_id_test_loader  Test ID Accuracy: {test_id_acc:.4f}')

# 调用函数绘制并保存对比图
plot_ecg_comparison(no_id_test_loader, test_loader, num_samples=50, save_path="ecg_comparison_mit_noise_平方_不是clamp_范式平滑noise_limit=1.0.png")







def compute_pearson_correlation(loader1, loader2):
    correlations = []
    for (reconstructed, _, _), (original, _, _) in zip(loader1, loader2):
        for i in range(reconstructed.size(0)):
            corr = np.corrcoef(reconstructed[i].squeeze().cpu().numpy(), original[i].squeeze().cpu().numpy())[0, 1]
            correlations.append(corr)
    return np.mean(correlations)

pearson_corr = compute_pearson_correlation(no_id_test_loader, test_loader)
print(f"Average Pearson Correlation: {pearson_corr}")


def compute_rmse(loader1, loader2):
    rmse_values = []
    for (reconstructed, _, _), (original, _, _) in zip(loader1, loader2):
        for i in range(reconstructed.size(0)):
            rmse = np.sqrt(np.mean((reconstructed[i].squeeze().cpu().numpy() - original[i].squeeze().cpu().numpy()) ** 2))
            rmse_values.append(rmse)
    return np.mean(rmse_values)

rmse = compute_rmse(no_id_test_loader, test_loader)
print(f"Average RMSE: {rmse}")

from scipy.linalg import sqrtm


def compute_frechet_distance(loader1, loader2):
    reconstructed_features = []
    original_features = []

    for (reconstructed, _, _), (original, _, _) in zip(loader1, loader2):
        reconstructed_features.append(reconstructed.view(reconstructed.size(0), -1).cpu().numpy())
        original_features.append(original.view(original.size(0), -1).cpu().numpy())

    reconstructed_features = np.concatenate(reconstructed_features, axis=0)
    original_features = np.concatenate(original_features, axis=0)

    mu_reconstructed = np.mean(reconstructed_features, axis=0)
    mu_original = np.mean(original_features, axis=0)
    sigma_reconstructed = np.cov(reconstructed_features, rowvar=False)
    sigma_original = np.cov(original_features, rowvar=False)

    mean_diff = np.sum((mu_reconstructed - mu_original) ** 2)
    covmean = sqrtm(sigma_reconstructed.dot(sigma_original))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = mean_diff + np.trace(sigma_reconstructed + sigma_original - 2 * covmean)
    return fd


frechet_distance = compute_frechet_distance(no_id_test_loader, test_loader)
print(f"Frechet Distance: {frechet_distance}")


freq_importance_id, freq_edges = compute_frequency_importance(
    model_id, no_id_test_loader, fs=360, freq_bins=360, device=device
)
freq_importance_heart, _ = compute_frequency_importance(
    model_heart, no_id_test_loader, fs=360, freq_bins=360, device=device
)

print(f"Length of freq_edges: {len(freq_edges)}")
print(f"Length of freq_importance_id: {len(freq_importance_id)}")
print(f"Length of freq_importance_heart: {len(freq_importance_heart)}")

plot_frequency_heatmap(freq_importance_id, freq_importance_heart, freq_edges, save_path="no_id_test_loader_frequency_heatmap_importance_l2噪声noise_limit=1.0.png")

freq_guidance = compute_freq_guidance_mask(freq_importance_id, freq_importance_heart, device=device)

print(freq_guidance)


freq_distribution, freq_edges = compute_frequency_distribution(train_loader, fs=360, freq_bins=360, device='cuda')

plot_frequency_distribution(freq_distribution, freq_edges, loader_name="train_loader", save_path="freq_distribution_no_id_test_loader_l2噪声noise_limit=1.0 .png")


import matplotlib.pyplot as plt
import torch
import numpy as np

# 函数：累积时域和频域信号并绘制
def plot_loader_time_and_frequency(loader, loader_name, fs=360, save_path=None):
    """
    累积加载器中所有信号的时域和频域特性，并绘制图像。
    Args:
        loader: 数据加载器
        loader_name: 加载器名称（trainloader 或 noid testloader）
        fs: 采样频率
        save_path: 图像保存路径
    """
    # 累积时域信号和频域信号
    cumulative_time_signal = None
    cumulative_frequency_signal = None

    for inputs, _, _ in loader:
        # 转换为 numpy 并累积时域信号
        batch_signals = inputs.squeeze().cpu().numpy()  # 提取 batch 中所有信号
        if cumulative_time_signal is None:
            cumulative_time_signal = np.sum(batch_signals, axis=0)  # 初始化
        else:
            cumulative_time_signal += np.sum(batch_signals, axis=0)  # 累积

        # 转换到频域并累积
        batch_frequency = np.abs(np.fft.rfft(batch_signals, axis=1))  # 频域幅值
        if cumulative_frequency_signal is None:
            cumulative_frequency_signal = np.sum(batch_frequency, axis=0)  # 初始化
        else:
            cumulative_frequency_signal += np.sum(batch_frequency, axis=0)  # 累积

    # 时间轴和频率轴
    time_axis = np.linspace(0, len(cumulative_time_signal) / fs, len(cumulative_time_signal))
    freq_axis = np.fft.rfftfreq(len(cumulative_time_signal), d=1/fs)

    # 绘图
    plt.figure(figsize=(14, 6))

    # 绘制时域累积信号
    plt.subplot(1, 2, 1)
    plt.plot(time_axis, cumulative_time_signal)
    plt.title(f"{loader_name} - Cumulative Time Domain Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 绘制频域累积信号
    plt.subplot(1, 2, 2)
    plt.plot(freq_axis, cumulative_frequency_signal)
    plt.title(f"{loader_name} - Cumulative Frequency Domain Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# 对 trainloader 和 noid testloader 分别绘制
plot_loader_time_and_frequency(train_loader, "Train Loader", fs=360, save_path="cumulative_trainloader.png")
plot_loader_time_and_frequency(test_loader, "Test Loader", fs=360, save_path="cumulative_testloader.png")
plot_loader_time_and_frequency(no_id_test_loader, "NoID Test Loader", fs=360, save_path="cumulative_noid_testloader.png")



