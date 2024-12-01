# import wfdb
#
# # 下载MIT-BIH数据集  也有很多别的数据集 后面可以换
# wfdb.dl_database('mitdb', './data/raw/mit/')
#
import torch.nn.functional as F
import torch.nn.functional as F
import wfdb
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class PTBDataset(Dataset):
    def __init__(self, data_dir, segment_length=1000):
        self.data_dir = data_dir
        self.data = []  # 存储 ECG 片段
        self.ecg_labels = []  # 存储疾病标签
        self.id_labels = []  # 存储病人 ID 标签
        self.patient_id_map = {}  # 病人 ID 映射

        # 获取记录列表
        record_list = wfdb.get_record_list('ptbdb')
        unique_patients = sorted(set(record.split('/')[0] for record in record_list))
        self.patient_id_map = {patient_id: idx for idx, patient_id in enumerate(unique_patients)}

        # 遍历每条记录并提取 ECG 信号和标签
        for record_name in record_list:
            record_path = os.path.join(self.data_dir, record_name)
            header = wfdb.rdheader(record_path)
            diagnosis = self.get_ecg_label(header)
            patient_id = record_name.split('/')[0]
            id_label = self.patient_id_map[patient_id]

            # 跳过没有疾病标签的记录
            if diagnosis is None:
                print(f"Warning: No diagnosis found for {record_name}")
                continue

            record = wfdb.rdrecord(record_path)
            ecg_signal = record.p_signal[:, 0]  # 选取第一个通道的 ECG 信号

            # 将信号分段
            for i in range(0, len(ecg_signal) - segment_length, segment_length):
                segment = ecg_signal[i:i + segment_length]
                self.data.append(torch.tensor(segment).float())
                self.ecg_labels.append(diagnosis)  # 疾病标签
                self.id_labels.append(id_label)  # 病人 ID 标签

    def get_ecg_label(self, header):
        """根据头部注释提取疾病标签。"""
        # "diagnose": {
        #     "Myocardial infarction": 368,
        #     "Hypertrophy": 7,
        #     "Healthy control": 80,
        #     "n/a": 27,
        #     "Myocarditis": 4,
        #     "Stable angina": 2,
        #     "Cardiomyopathy": 17,
        #     "Unstable angina": 1,
        #     "Bundle branch block": 17,
        #     "Dysrhythmia": 16,
        #     "Valvular heart disease": 6,
        #     "Heart failure (NYHA 3)": 1,
        #     "Heart failure (NYHA 2)": 1,
        #     "Palpitation": 1,
        #     "Heart failure (NYHA 4)": 1,
        #     "total:": "15 diagnose classes"
        # },

        label_map = {
            "Myocardial infarction": 0,
            "Cardiomyopathy": 1,
            # "Valvular heart disease": 2,
            # "Dysrhythmia": 3,
            # "Hypertrophy": 4,
            # "Myocarditis": 5,
            # "Stable angina": 6,
            # "Unstable angina": 7,
            # "Heart failure (NYHA 2)": 8,
            # "Heart failure (NYHA 3)": 9,
            # "Heart failure (NYHA 4)": 10,
            # "Healthy control": 11,
            # "n/a": 12,  # 如果你想忽略“n/a”，可以删除这一行

        }

        for comment in header.comments:
            for label, value in label_map.items():
                if label in comment:
                    return value  # 返回疾病标签
        return None  # 没有疾病标签则返回None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].unsqueeze(0)  # 增加一个通道维度
        ecg_label = torch.tensor(self.ecg_labels[idx], dtype=torch.long)  # 疾病标签
        id_label = torch.tensor(self.id_labels[idx], dtype=torch.long)  # 病人 ID 标签
        return sample, ecg_label, id_label


# 创建数据集实例
data_dir = './data/raw/ptb/'  # 假设数据存储在这个目录下
dataset = PTBDataset(data_dir)

# 从数据集中提取样本
sample, heart_label, id_label = dataset[0]

print(f"Sample shape: {sample.shape}")
print(f"Heart label: {heart_label}")
print(f"Patient ID label: {id_label}")

# 统计总样本数量
total_samples = len(dataset)
print(f"Total number of samples: {total_samples}")



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

        self.fc1 = nn.Linear(15872, 128)
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

        self.fc1 = nn.Linear(15872, 128)
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
            inputs, heart_labels = inputs.to(device), heart_labels.to(device)

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

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.8f}, Accuracy: {accuracy:.8f}, '
              f'Val Loss: {val_loss:.8f}, Val Accuracy: {val_acc:.8f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 保存最佳模型到文件
    torch.save(model.state_dict(), 'model_heart_ptb.pth')
    print("最佳心跳分类模型已保存为 model_heart_ptb.pth")


def train_id_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    best_val_loss = float('inf')
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        best_val_acc= 0.0
        correct = 0
        total = 0
        best_acc = 0

        for inputs, _, id_labels in train_loader:  # 只取ID分类标签
            inputs, id_labels = inputs.to(device), id_labels.to(device)

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
        if val_acc> best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.8f}, Accuracy: {accuracy:.4f}, '
              f'Val Loss: {val_loss:.8f}, Val Accuracy: {val_acc:.8f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 保存最佳模型到文件
    torch.save(model.state_dict(), 'model_id_ptb.pth')
    print("最佳病人ID分类模型已保存为 model_id_ptb.pth")


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
# train_indices, temp_indices = train_test_split(indices, train_size=train_size, random_state=42, shuffle=False)
# val_indices, test_indices = train_test_split(temp_indices, train_size=val_size / (val_size + test_size),
#                                              random_state=42,shuffle=False)

# 使用train_test_split划分训练集、验证集和测试集
train_indices, temp_indices = train_test_split(indices, train_size=train_size, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, train_size=val_size / (val_size + test_size),
                                             random_state=42)

# 创建Subset和DataLoader
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义损失函数和优化器
heart_criterion = nn.CrossEntropyLoss()
id_criterion = nn.CrossEntropyLoss()


# Before training, print the unique labels to check their range
print(f"Unique heart labels: {torch.unique(torch.tensor([label for _, label, _ in dataset]))}")
print(f"Unique id labels: {torch.unique(torch.tensor([id_label for _, _, id_label in dataset]))}")


heart_model = HeartbeatCNN(num_classes=2).to(device)
heart_optimizer = optim.Adam(heart_model.parameters(), lr=0.001)
train_heart_model(heart_model, train_loader, val_loader, heart_criterion, heart_optimizer, num_epochs=200, device=device)

# 实例化病人ID分类模型并训练
num_id_classes = len(dataset.patient_id_map)  # 患者ID的唯一数量
id_model = PatientIDCNN(num_classes=num_id_classes).to(device)
id_optimizer = optim.Adam(id_model.parameters(), lr=0.001)
train_id_model(id_model, train_loader, val_loader, id_criterion, id_optimizer, num_epochs=300, device=device)


# 加载心跳分类模型
model_heart = HeartbeatCNN(num_classes=2).to(device)
model_heart.load_state_dict(torch.load('model_heart_ptb.pth'))

# 加载病人ID分类模型
num_id_classes = len(dataset.patient_id_map)  # 患者ID的唯一数量
model_id = PatientIDCNN(num_classes=num_id_classes).to(device)
model_id.load_state_dict(torch.load('model_id_ptb.pth'))

# 在测试集上评估心跳分类模型
test_heart_loss, test_heart_acc = evaluate_heart_model(model_heart, test_loader, heart_criterion, device=device)
print(f'Test Heart Loss: {test_heart_loss:.4f}, Test Heart Accuracy: {test_heart_acc:.4f}')

# 在测试集上评估病人ID分类模型
test_id_loss, test_id_acc = evaluate_id_model(model_id, test_loader, id_criterion, device=device)
print(f'Test ID Loss: {test_id_loss:.4f}, Test ID Accuracy: {test_id_acc:.4f}')





import matplotlib.pyplot as plt
import random
import torch


def plot_ecg_comparison(loader1, loader2, num_samples=50, save_path="ecg_comparison.png"):
    # 随机选择50个样本的索引
    indices = random.sample(range(len(loader1.dataset)), num_samples)
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

import torch
import torch.nn as nn





import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomUNetWithDynamicRange(nn.Module):
    def __init__(self, input_dim, global_min=-0.01, global_max=0.01):
        super(CustomUNetWithDynamicRange, self).__init__()
        self.global_min = global_min  # 全局最小噪声范围
        self.global_max = global_max  # 全局最大噪声范围

        # 编码器
        self.encoder1 = nn.Conv1d(1, 8, kernel_size=5, stride=5)
        self.skip1 = nn.Conv1d(8, 1, kernel_size=1)

        self.encoder2 = nn.Conv1d(8, 32, kernel_size=3, stride=3)
        self.skip2 = nn.Conv1d(32, 1, kernel_size=1)

        self.encoder3 = nn.Conv1d(32, 3, kernel_size=8, stride=8)

        # 解码器
        self.decoder3 = nn.ConvTranspose1d(3, 32, kernel_size=8, stride=8)
        self.upskip3 = nn.Conv1d(3, 32, kernel_size=1)

        self.decoder2 = nn.ConvTranspose1d(32, 8, kernel_size=3, stride=3)
        self.upskip2 = nn.Conv1d(1, 8, kernel_size=1)

        self.decoder1 = nn.ConvTranspose1d(8, 1, kernel_size=5, stride=5)
        self.upskip1 = nn.Conv1d(1, 1, kernel_size=1)

        # 噪声范围预测模块
        self.noise_range_predictor = nn.Conv1d(1, 2, kernel_size=1)

    def forward(self, x):
        # 编码阶段
        e1 = self.encoder1(x)
        s1 = self.skip1(e1)

        e2 = self.encoder2(e1)
        s2 = self.skip2(e2)

        e3 = self.encoder3(e2)
        s3 = e3

        # 解码阶段
        d3 = self.decoder3(e3)
        d3 = d3 + F.interpolate(self.upskip3(s3), size=d3.size(2), mode='linear', align_corners=True)

        d2 = self.decoder2(d3)
        d2 = d2 + F.interpolate(self.upskip2(s2), size=d2.size(2), mode='linear', align_corners=True)

        d1 = self.decoder1(d2)
        d1 = d1 + F.interpolate(self.upskip1(s1), size=d1.size(2), mode='linear', align_corners=True)

        # 将解码器输出调整为与输入相同的长度 F.interpolate 会通过插值来增加长度（类似于“填充”）  F.interpolate 会通过插值减少长度（类似于“裁剪”）
        d1 = F.interpolate(d1, size=x.size(2), mode='linear', align_corners=True)

        # 噪声计算
        noise = d1 - x

        # 动态范围预测
        noise_range = self.noise_range_predictor(x)
        min_noise_map = torch.clamp(noise_range[:, 0, :].unsqueeze(1), min=self.global_min, max=self.global_max)
        max_noise_map = torch.clamp(noise_range[:, 1, :].unsqueeze(1), min=self.global_min, max=self.global_max)

        # 对噪声进行局部和全局裁剪
        noise = torch.max(torch.min(noise, max_noise_map), min_noise_map)
        return noise


class NoiseGenerator(nn.Module):
    #初始化满足约束
    def __init__(self, input_dim):
        super(NoiseGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.min_noise = -0.008  # 设置最小噪声值
        self.max_noise = 0.008   # 设置最大噪声值

    def forward(self, x):
        noise = self.fc(x)
        # 裁剪噪声范围
        noise = torch.clamp(noise, min=self.min_noise, max=self.max_noise)
        return noise


import matplotlib.pyplot as plt

def train_noise_generator_with_logging(generator, train_loader, heart_model, id_model, optimizer, num_epochs=20, device='cpu'):
    """
    Train the noise generator and log detailed loss and weight changes.
    """
    heart_model.eval()  # Fix heart classification model
    id_model.eval()  # Fix patient ID classification model

    for param in heart_model.parameters():
        param.requires_grad = False
    for param in id_model.parameters():
        param.requires_grad = False

    # Create dictionaries to store loss and weight values
    loss_logs = {
        'epoch': [],
        'heart_loss': [],
        'anti_id_loss': [],
        'id_to_heart_loss': [],
        'similarity_loss': [],
        'noise_regularization': [],
        'total_loss': [],
    }
    weight_logs = {
        'heart_loss': [],
        'anti_id_loss': [],
        'id_to_heart_loss': [],
        'similarity_loss': [],
        'noise_regularization': [],
    }

    # Initialize weights
    weights = {
        'heart_loss': 1.0,
        'anti_id_loss': 10,
        'id_to_heart_loss': 1,
        'similarity_loss': 1.0,
        'noise_regularization': 1.0,
    }

    for epoch in range(num_epochs):
        generator.train()
        epoch_heart_loss = 0.0
        epoch_anti_id_loss = 0.0
        epoch_id_to_heart_loss = 0.0
        epoch_similarity_loss = 0.0
        epoch_noise_regularization = 0.0
        epoch_total_loss = 0.0

        for data, heart_labels, id_labels in train_loader:
            data, heart_labels, id_labels = data.to(device), heart_labels.to(device), id_labels.to(device)

            optimizer.zero_grad()

            # Generate noise and create perturbed data
            noise = generator(data)
            perturbed_data = data + noise

            # Compute individual losses
            heart_outputs = heart_model(perturbed_data)
            heart_loss = nn.CrossEntropyLoss()(heart_outputs, heart_labels)

            id_outputs = id_model(perturbed_data)
            anti_id_loss = -nn.CrossEntropyLoss()(id_outputs, id_labels)
            id_to_heart_loss = nn.CrossEntropyLoss()(id_outputs, heart_labels)

            similarity_loss = F.mse_loss(perturbed_data, data)
            noise_regularization = torch.mean(noise ** 2)

            # Adjust weights dynamically (every 10 epochs)
            # if epoch % 10 == 0 and epoch > 0:
                # weights['heart_loss'] *= 1.1
                # weights['anti_id_loss'] *= 1.1


            # Compute total weighted loss
            total_loss = (
                weights['heart_loss'] * heart_loss +
                weights['anti_id_loss'] * anti_id_loss +
                weights['id_to_heart_loss'] * id_to_heart_loss +
                weights['similarity_loss'] * similarity_loss +
                weights['noise_regularization'] * noise_regularization
            )

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            # Accumulate epoch losses
            epoch_heart_loss += heart_loss.item()
            epoch_anti_id_loss += anti_id_loss.item()
            epoch_id_to_heart_loss += 0.01*id_to_heart_loss.item()
            epoch_similarity_loss += similarity_loss.item()
            epoch_noise_regularization += noise_regularization.item()
            epoch_total_loss += total_loss.item()

        # Log average losses for the epoch
        num_batches = len(train_loader)
        loss_logs['epoch'].append(epoch + 1)
        loss_logs['heart_loss'].append(epoch_heart_loss / num_batches)
        loss_logs['anti_id_loss'].append(epoch_anti_id_loss / num_batches)
        loss_logs['id_to_heart_loss'].append(epoch_id_to_heart_loss / num_batches)
        loss_logs['similarity_loss'].append(epoch_similarity_loss / num_batches)
        loss_logs['noise_regularization'].append(epoch_noise_regularization / num_batches)
        loss_logs['total_loss'].append(epoch_total_loss / num_batches)

        # Log weights for the epoch
        for key in weights.keys():
            weight_logs[key].append(weights[key])

        # Print the current epoch's losses and weights
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Heart Loss: {loss_logs['heart_loss'][-1]:.10f}")
        print(f"  Anti-ID Loss: {loss_logs['anti_id_loss'][-1]:.10f}")
        print(f"  ID-to-Heart Noise Loss: {loss_logs['id_to_heart_loss'][-1]:.10f}")
        print(f"  Similarity Loss: {loss_logs['similarity_loss'][-1]:.10f}")
        print(f"  Noise Regularization: {loss_logs['noise_regularization'][-1]:.10f}")
        print(f"  Total Loss: {loss_logs['total_loss'][-1]:.10f}")
        print(f"  Current Weights: {weights}")

    return loss_logs, weight_logs


def plot_losses_and_weights(loss_logs, weight_logs, save_path='loss_weight_analysis.png'):
    """
    Plot loss metrics and weight changes over epochs.
    """
    epochs = loss_logs['epoch']

    plt.figure(figsize=(16, 10))

    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_logs['heart_loss'], label='Heart Loss', marker='o')
    plt.plot(epochs, loss_logs['anti_id_loss'], label='Anti-ID Loss', marker='x')
    plt.plot(epochs, loss_logs['id_to_heart_loss'], label='ID-to-Heart Noise Loss', marker='^')
    plt.plot(epochs, loss_logs['similarity_loss'], label='Similarity Loss', marker='s')
    plt.plot(epochs, loss_logs['noise_regularization'], label='Noise Regularization', marker='*')
    plt.plot(epochs, loss_logs['total_loss'], label='Total Loss', linestyle='--', marker='v')
    plt.title('Loss Convergence During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)

    # Plot weights
    plt.subplot(2, 1, 2)
    for key in weight_logs.keys():
        plt.plot(epochs, weight_logs[key], label=f'{key} Weight', linestyle='-', marker='o')
    plt.title('Weight Changes During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss and weight curve saved to {save_path}")




def create_noisy_test_loader(generator, test_loader, device):
    """
    使用训练好的噪声生成器生成带噪的测试数据加载器。
    """
    generator.eval()  # 固定噪声生成器参数
    noisy_data = []
    heart_labels = []
    id_labels = []

    with torch.no_grad():
        for data, heart_label, id_label in test_loader:
            data = data.to(device)
            noise = generator(data)  # 生成噪声
            perturbed_data = data + noise  # 添加噪声

            noisy_data.append(perturbed_data.cpu())
            heart_labels.append(heart_label)
            id_labels.append(id_label)

    # 创建新的 DataLoader
    noisy_dataset = torch.utils.data.TensorDataset(torch.cat(noisy_data), torch.cat(heart_labels), torch.cat(id_labels))
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=test_loader.batch_size, shuffle=False)
    return noisy_loader



# # 实例化噪声生成器
# input_dim = 360 # ECG 信号的长度
# generator = NoiseGenerator(input_dim=input_dim).to(device)
# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 实例化噪声生成器
input_dim = 1000  # ECG 信号的长度  1000---360
# generator = CustomUNetWithDynamicRange(input_dim).to(device)
generator = NoiseGenerator(input_dim).to(device)
# optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
# 训练噪声生成器并记录损失
loss_logs,weight_logs = train_noise_generator_with_logging(generator, train_loader, model_heart, model_id, optimizer, num_epochs=50, device=device)

plot_losses_and_weights(loss_logs, weight_logs, save_path='loss_and_weight_analysis_ptb.png')


# 使用噪声生成器生成带噪测试数据
no_id_test_loader = create_noisy_test_loader(generator, test_loader, device)

import torch

def save_noisy_loader(noisy_loader, file_path):
    """
    将带噪数据保存到文件。
    Args:
        noisy_loader: 带噪的 DataLoader。
        file_path: 保存文件的路径。
    """
    noisy_data = []
    heart_labels = []
    id_labels = []

    # 将数据提取并保存到列表
    for inputs, heart_label, id_label in noisy_loader:
        noisy_data.append(inputs.cpu())  # 确保数据在 CPU 上
        heart_labels.append(heart_label.cpu())
        id_labels.append(id_label.cpu())

    # 将数据保存到文件
    torch.save({
        'noisy_data': torch.cat(noisy_data),
        'heart_labels': torch.cat(heart_labels),
        'id_labels': torch.cat(id_labels)
    }, file_path)

    print(f"Noisy loader saved to {file_path}")

def load_noisy_loader(file_path, batch_size, device='cpu'):
    """
    从文件加载带噪数据并创建 DataLoader。
    Args:
        file_path: 保存文件的路径。
        batch_size: DataLoader 的批量大小。
        device: 数据加载到的设备。
    Returns:
        DataLoader: 加载后的 DataLoader。
    """
    data = torch.load(file_path)
    noisy_data = data['noisy_data'].to(device, dtype=torch.float32)
    heart_labels = data['heart_labels'].to(device)
    id_labels = data['id_labels'].to(device)

    # 创建 TensorDataset 和 DataLoader
    noisy_dataset = torch.utils.data.TensorDataset(noisy_data, heart_labels, id_labels)
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False)

    print(f"Noisy loader loaded from {file_path}")
    return noisy_loader


noisy_loader_file = "noisy_test_loader_ptb.pt"

# 保存带噪数据
save_noisy_loader(no_id_test_loader, noisy_loader_file)
##读取
batch_size = test_loader.batch_size

# 加载带噪数据
no_id_test_loader = load_noisy_loader(noisy_loader_file, batch_size=batch_size, device=device)

# 检查加载的数据
for inputs, heart_labels, id_labels in no_id_test_loader:
    print(f"Loaded noisy inputs shape: {inputs.shape}")
    print(f"Heart labels shape: {heart_labels.shape}")
    print(f"ID labels shape: {id_labels.shape}")
    break

# 加载心跳分类模型
# model_heart = HeartbeatCNN(num_classes=2).to(device)
model_heart.load_state_dict(torch.load('model_heart_ptb.pth'))

# # 加载病人ID分类模型
# num_id_classes = len(dataset.patient_id_map)  # 患者ID的唯一数量
# model_id = PatientIDCNN(num_classes=num_id_classes).to(device)
model_id.load_state_dict(torch.load('model_id_ptb.pth'))


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
plot_ecg_comparison(no_id_test_loader, test_loader, num_samples=50, save_path="ecg_comparison_ptb_noise.png")


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


# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, lfilter
# from torch.utils.data import DataLoader, TensorDataset
# import torch
# def analyze_frequency_difference(no_id_loader, original_loader, fs=1000):
#     """
#     Analyze frequency differences between two loaders (no ID vs. original).
#     Args:
#         no_id_loader: DataLoader with no ID information.
#         original_loader: DataLoader with original data.
#         fs: Sampling frequency (Hz).
#     Returns:
#         mean_freq_diff: Average magnitude difference in frequency domain.
#         freqs: Frequency values (Hz).
#     """
#     freq_differences = []
#
#     for (noid_data, _, _), (orig_data, _, _) in zip(no_id_loader, original_loader):
#         # Calculate FFT for each signal in the batch
#         for i in range(noid_data.size(0)):
#             noid_fft = np.fft.fft(noid_data[i].squeeze().cpu().numpy())
#             orig_fft = np.fft.fft(orig_data[i].squeeze().cpu().numpy())
#             diff = np.abs(noid_fft - orig_fft)  # Magnitude difference
#             freq_differences.append(diff)
#
#     # Average the differences across all samples
#     mean_freq_diff = np.mean(freq_differences, axis=0)
#     freqs = np.fft.fftfreq(len(mean_freq_diff), d=1 / fs)
#
#     return mean_freq_diff, freqs
#
# def plot_and_save_frequency_analysis(mean_freq_diff, freqs, save_path='frequency_difference.png'):
#     """
#     Plot and save the frequency analysis result.
#     Args:
#         mean_freq_diff: Average magnitude difference in frequency domain.
#         freqs: Frequency values (Hz).
#         save_path: Path to save the plot.
#     """
#     # Only keep positive frequencies
#     positive_freqs = freqs[freqs >= 0]
#     positive_magnitude = mean_freq_diff[freqs >= 0]
#
#     # Plot the frequency spectrum
#     plt.figure(figsize=(10, 6))
#     plt.plot(positive_freqs, positive_magnitude, label='Frequency Difference')
#     plt.title("Frequency Difference (No ID vs. Original)")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Difference Magnitude")
#     plt.grid()
#     plt.legend()
#     plt.savefig(save_path)  # Save the plot
#     plt.show()
#     print(f"Frequency analysis plot saved to {save_path}")
#
# mean_freq_diff, freqs = analyze_frequency_difference(no_id_test_loader, test_loader)
# plot_and_save_frequency_analysis(mean_freq_diff, freqs, save_path='frequency_difference_analysis.png')
#
#
# def analyze_frequency_comparison(test_loader, no_id_test_loader, fs=1000, save_path="frequency_comparison_distribution.png"):
#     """
#     分析原始 ECG 信号和带噪 ECG 信号的频率分布，并绘制在同一张图上。
#
#     Args:
#         test_loader: DataLoader, 原始 ECG 测试集加载器。
#         no_id_test_loader: DataLoader, 带噪声的 ECG 测试集加载器。
#         fs: 采样频率（默认 1000 Hz）。
#         save_path: 图像保存路径。
#     """
#     def compute_average_frequency(loader):
#         magnitudes = []  # 存储所有信号的频谱幅值
#
#         for inputs, _, _ in loader:
#             inputs = inputs.cpu().numpy()  # 将张量转换为 NumPy 数组
#             for signal in inputs:
#                 # 对每条信号计算 FFT 并取绝对值
#                 fft_result = np.fft.fft(signal.squeeze())
#                 magnitudes.append(np.abs(fft_result))
#
#         # 计算平均频谱
#         avg_magnitude = np.mean(magnitudes, axis=0)
#         freqs = np.fft.fftfreq(len(avg_magnitude), d=1 / fs)  # 频率范围
#
#         # 只保留正频率部分
#         positive_freqs = freqs[freqs >= 0]
#         positive_magnitude = avg_magnitude[freqs >= 0]
#
#         return positive_freqs, positive_magnitude
#
#     # 分别计算两组信号的平均频谱
#     test_freqs, test_magnitude = compute_average_frequency(test_loader)
#     noid_freqs, noid_magnitude = compute_average_frequency(no_id_test_loader)
#
#     # 绘制频谱图
#     plt.figure(figsize=(10, 6))
#     plt.plot(test_freqs, test_magnitude, label='Original ECG Frequency Spectrum', color='blue')
#     plt.plot(noid_freqs, noid_magnitude, label='Noisy ECG Frequency Spectrum', color='red', linestyle='--')
#     plt.title("Frequency Distribution Comparison: Original vs Noisy ECG")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")
#     plt.grid(True)
#     plt.legend()
#
#     # 保存图像
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Frequency comparison plot saved to {save_path}")
#
#
# # 调用函数对两组信号进行频率分析并绘制比较图
# analyze_frequency_comparison(test_loader, no_id_test_loader, fs=1000, save_path="frequency_comparison_distribution.png")
# import numpy as np
# import matplotlib.pyplot as plt
#
# def plot_2d_frequency_distribution(test_loader, no_id_test_loader, fs=1000, save_path="2d_frequency_comparison.png"):
#     """
#     绘制原始信号和加噪信号的二维频率分布图（直方图）。
#     Args:
#         test_loader: DataLoader, 原始 ECG 测试集加载器。
#         no_id_test_loader: DataLoader, 加噪 ECG 测试集加载器。
#         fs: 采样频率（默认 1000 Hz）。
#         save_path: 图像保存路径。
#     """
#     def compute_frequency_matrix(loader):
#         """
#         计算每条信号的频率幅值，返回频率矩阵。
#         """
#         frequency_matrix = []
#         for inputs, _, _ in loader:
#             inputs = inputs.cpu().numpy()
#             for signal in inputs:
#                 fft_result = np.fft.fft(signal.squeeze())
#                 frequency_matrix.append(np.abs(fft_result[:len(fft_result) // 2]))  # 只保留正频率部分
#         return np.array(frequency_matrix)
#
#     # 计算原始信号和加噪信号的频率矩阵
#     original_freq_matrix = compute_frequency_matrix(test_loader)
#     noisy_freq_matrix = compute_frequency_matrix(no_id_test_loader)
#
#     # 计算平均频率差异矩阵
#     difference_matrix = noisy_freq_matrix - original_freq_matrix
#
#     # 绘制二维热力图
#     plt.figure(figsize=(12, 8))
#     plt.imshow(
#         difference_matrix.T,
#         aspect='auto',
#         origin='lower',
#         cmap='coolwarm',
#         extent=[0, len(difference_matrix), 0, fs // 2]
#     )
#     plt.colorbar(label='Magnitude Difference')
#     plt.title("2D Frequency Distribution Comparison: Original vs Noisy ECG")
#     plt.xlabel("Signal Index")
#     plt.ylabel("Frequency (Hz)")
#     plt.savefig(save_path)
#     plt.close()
#     print(f"2D frequency distribution plot saved to {save_path}")
#
# # 调用函数生成并保存二维频率分布对比图
# plot_2d_frequency_distribution(test_loader, no_id_test_loader, fs=1000, save_path="2d_frequency_comparison.png")
# import numpy as np
# import matplotlib.pyplot as plt
#
# def plot_dual_frequency_distribution_with_scaling(
#     test_loader, no_id_test_loader, fs=1000, save_path="dual_frequency_comparison_log.png"):
#     """
#     绘制原始信号和加噪信号的二维频率分布图，加入对数缩放和归一化处理。
#     """
#     def compute_frequency_matrix(loader):
#         """
#         计算每条信号的频率幅值，返回频率矩阵。
#         """
#         frequency_matrix = []
#         for inputs, _, _ in loader:
#             inputs = inputs.cpu().numpy()
#             for signal in inputs:
#                 fft_result = np.fft.fft(signal.squeeze())
#                 frequency_matrix.append(np.abs(fft_result[:len(fft_result) // 2]))  # 只保留正频率部分
#         return np.array(frequency_matrix)
#
#     # 计算频率矩阵
#     original_freq_matrix = compute_frequency_matrix(test_loader)
#     noisy_freq_matrix = compute_frequency_matrix(no_id_test_loader)
#
#     # 对数缩放
#     log_original_freq_matrix = np.log1p(original_freq_matrix)  # log(1+x) 避免 log(0) 问题
#     log_noisy_freq_matrix = np.log1p(noisy_freq_matrix)
#
#     # 绘制双子图
#     fig, axes = plt.subplots(1, 2, figsize=(16, 8))
#
#     # 原始信号频率分布
#     im1 = axes[0].imshow(
#         log_original_freq_matrix.T,
#         aspect='auto',
#         origin='lower',
#         cmap='viridis',
#         extent=[0, len(log_original_freq_matrix), 0, fs // 2]
#     )
#     axes[0].set_title("Original ECG Frequency Distribution (Log Scale)")
#     axes[0].set_xlabel("Signal Index")
#     axes[0].set_ylabel("Frequency (Hz)")
#     fig.colorbar(im1, ax=axes[0])
#
#     # 加噪信号频率分布
#     im2 = axes[1].imshow(
#         log_noisy_freq_matrix.T,
#         aspect='auto',
#         origin='lower',
#         cmap='viridis',
#         extent=[0, len(log_noisy_freq_matrix), 0, fs // 2]
#     )
#     axes[1].set_title("Noisy ECG Frequency Distribution (Log Scale)")
#     axes[1].set_xlabel("Signal Index")
#     axes[1].set_ylabel("Frequency (Hz)")
#     fig.colorbar(im2, ax=axes[1])
#
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Dual frequency distribution plot saved to {save_path}")
#
# # 调用函数生成并保存双子图
# plot_dual_frequency_distribution_with_scaling(test_loader, no_id_test_loader, fs=1000, save_path="dual_frequency_comparison.png")

#
# import numpy as np
# import matplotlib.pyplot as plt
#
# def sliding_window_fft(loader, fs=1000, window_size=256, step_size=128):
#     """
#     对 DataLoader 中的信号进行滑动窗口 FFT 分析。
#     Args:
#         loader: DataLoader, 包含信号数据。
#         fs: 采样频率（默认 1000 Hz）。
#         window_size: 窗口大小（样本点数）。
#         step_size: 滑动步长（样本点数）。
#     Returns:
#         frequency_results: List, 包含每个信号的频谱矩阵。
#     """
#     frequency_results = []
#
#     for inputs, _, _ in loader:
#         inputs = inputs.cpu().numpy()  # 转换为 NumPy 数组
#         for signal in inputs:
#             signal = signal.squeeze()
#             num_windows = (len(signal) - window_size) // step_size + 1
#             freq_magnitudes = []
#
#             # 滑动窗口 FFT
#             for i in range(num_windows):
#                 start = i * step_size
#                 end = start + window_size
#                 window_signal = signal[start:end]
#                 fft_result = np.fft.fft(window_signal)
#                 magnitudes = np.abs(fft_result[:window_size // 2])  # 只保留正频率
#                 freq_magnitudes.append(magnitudes)
#
#             frequency_results.append(np.array(freq_magnitudes))
#
#     return frequency_results
#
# def plot_sliding_fft_results(results, fs=1000, save_path="sliding_fft_analysis.png"):
#     """
#     绘制滑动窗口 FFT 结果的二维热力图。
#     Args:
#         results: 滑动窗口 FFT 结果列表。
#         fs: 采样频率。
#         save_path: 保存路径。
#     """
#     for i, freq_matrix in enumerate(results):
#         plt.figure(figsize=(10, 6))
#         plt.imshow(freq_matrix.T, aspect='auto', origin='lower',
#                    extent=[0, freq_matrix.shape[0], 0, fs // 2], cmap='viridis')
#         plt.colorbar(label='Magnitude')
#         plt.title(f"Sliding Window FFT - Signal {i}")
#         plt.xlabel("Window Index")
#         plt.ylabel("Frequency (Hz)")
#         plt.savefig(f"{save_path}_signal_{i}.png")
#         plt.close()
#
#         print(f"Saved sliding FFT plot for Signal {i} to {save_path}_signal_{i}.png")
#
# # 参数设置
# fs = 1000  # 采样频率
# window_size = 256  # 滑动窗口大小
# step_size = 128  # 滑动步长
#
# # 对 test_loader 和 no_id_test_loader 进行滑动窗口 FFT 分析
# test_fft_results = sliding_window_fft(test_loader, fs, window_size, step_size)
# noid_fft_results = sliding_window_fft(no_id_test_loader, fs, window_size, step_size)
#
# # 绘制并保存结果
# plot_sliding_fft_results(test_fft_results, fs, save_path="test_loader_fft")
# plot_sliding_fft_results(noid_fft_results, fs, save_path="noid_loader_fft")

def analyze_frequency_content(loader, fs=1000):
    """
    分析信号的平均频谱和主要能量分布。
    Args:
        loader: DataLoader
        fs: 采样频率
    Returns:
        avg_spectrum: 平均频谱
        freqs: 对应的频率范围
    """
    spectra = []
    for data, _, _ in loader:
        data = data.cpu().numpy()
        for signal in data:
            # Compute FFT and take absolute values
            fft_result = np.fft.fft(signal.squeeze())
            magnitudes = np.abs(fft_result[:len(fft_result) // 2])  # Only positive frequencies
            spectra.append(magnitudes)

    avg_spectrum = np.mean(spectra, axis=0)
    freqs = np.fft.fftfreq(len(signal.squeeze()), d=1 / fs)[:len(avg_spectrum)]  # Correct length
    return avg_spectrum, freqs


# 分析 test_loader 和 no_id_test_loader 的频率分布
avg_spectrum_test, freqs = analyze_frequency_content(test_loader, fs=1000)
avg_spectrum_noid, _ = analyze_frequency_content(no_id_test_loader, fs=1000)
# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(freqs, avg_spectrum_test, label="Original Signal (test_loader)", color='blue')
plt.plot(freqs, avg_spectrum_noid, label="Noisy Signal (no_id_test_loader)", color='red', linestyle='--')
plt.title("Average Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
plt.show()
plt.savefig('analyze_frequency_content.png')

def compute_energy_in_ranges(spectrum, freqs, freq_ranges):
    """
    计算频谱在指定频率范围内的能量。
    Args:
        spectrum: 频谱数据
        freqs: 频率范围
        freq_ranges: [(low, high), ...] 频率范围列表
    Returns:
        energy_dict: 每个频率范围内的能量字典
    """
    energy_dict = {}
    for low, high in freq_ranges:
        mask = (freqs >= low) & (freqs < high)
        energy = np.sum(spectrum[mask])  # 使用掩码对频谱能量求和
        energy_dict[f"{low}-{high}Hz"] = energy
    return energy_dict

# 定义频率范围
freq_ranges = [(0, 10), (10, 50), (50, 100), (100, 150), (150, 250),(250, 500)]

# 分析 test_loader 和 no_id_test_loader 的频段能量
energy_test = compute_energy_in_ranges(avg_spectrum_test, freqs, freq_ranges)
energy_noid = compute_energy_in_ranges(avg_spectrum_noid, freqs, freq_ranges)

# 输出结果
print("Energy Distribution (test_loader):", energy_test)
print("Energy Distribution (no_id_test_loader):", energy_noid)

from scipy.signal import butter, filtfilt
# 带通滤波器
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    实现带通滤波器
    Args:
        signal: 输入信号
        lowcut: 低频截止频率
        highcut: 高频截止频率
        fs: 采样频率
        order: 滤波器阶数
    Returns:
        filtered_signal: 滤波后的信号
    """
    nyquist = 0.5 * fs  # 奈奎斯特频率
    if lowcut >= highcut or highcut > nyquist or lowcut < 0:
        raise ValueError(f"Invalid filter range: {lowcut}-{highcut} Hz")

    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


# 陷波滤波器
def notch_filter(signal, notch_freq, fs, bandwidth=1):
    """
    实现带阻滤波器（陷波滤波器）。
    Args:
        signal: 输入信号
        notch_freq: 陷波频率
        fs: 采样频率
        bandwidth: 陷波带宽
    Returns:
        filtered_signal: 滤波后的信号
    """
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low = max((notch_freq - bandwidth / 2) / nyquist, 0)  # 确保 low >= 0
    high = min((notch_freq + bandwidth / 2) / nyquist, 1)  # 确保 high <= 1

    if low >= high:
        print(f"Warning: Invalid notch frequency range for {notch_freq} Hz. Skipping.")
        return signal  # 如果范围无效，返回原始信号

    b, a = butter(2, [low, high], btype='bandstop')  # 带阻滤波器
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# 更新的去噪函数
def denoise_loader(loader, fs, lowcut, highcut, noise_freqs, bandwidth=2):
    """
    对 DataLoader 中的所有信号进行去噪处理。
    Args:
        loader: DataLoader
        fs: 采样频率
        lowcut: 带通滤波器低频截止
        highcut: 带通滤波器高频截止
        noise_freqs: 陷波滤波器目标频率列表
        bandwidth: 陷波带宽
    Returns:
        denoised_signals: 去噪后的信号列表
    """
    denoised_signals = []
    for data, _, _ in loader:
        data = data.cpu().numpy()
        for signal in data:
            # 第一步：带通滤波
            filtered_signal = bandpass_filter(signal.squeeze(), lowcut, highcut, fs)

            # 第二步：逐个频段应用陷波滤波器
            for notch_freq in noise_freqs:
                filtered_signal = notch_filter(filtered_signal, notch_freq, fs, bandwidth)

            denoised_signals.append(filtered_signal)
    return denoised_signals

# 示例：对 no_id_test_loader 应用去噪
fs = 1000  # 采样频率
lowcut = 0.5  # 带通滤波器低频截止
highcut = 50  # 带通滤波器高频截止
noise_freqs = [50, 250, 500]  # 陷波滤波器目标频率

# 对 no_id_test_loader 去噪
denoised_no_id_signals = denoise_loader(no_id_test_loader, fs=1000, lowcut=0.5, highcut=50, noise_freqs=[50, 250], bandwidth=2)
from torch.utils.data import DataLoader, TensorDataset
def create_denoised_loader(denoised_signals, original_loader):
    """
    将去噪信号列表转换为 DataLoader。
    Args:
        denoised_signals: 去噪后的信号列表。
        original_loader: 原始 DataLoader，用于获取标签。
    Returns:
        DataLoader: 包含去噪信号及其标签的新 DataLoader。
    """
    # denoised_data = torch.tensor(denoised_signals).unsqueeze(1)  # 转换为张量并增加通道维度
    denoised_data = torch.tensor(denoised_signals, dtype=torch.float32).unsqueeze(1)  # 转换为 float32 张量并增加通道维度
    heart_labels = []
    id_labels = []

    # 提取原始标签
    for _, heart_label, id_label in original_loader:
        heart_labels.append(heart_label)
        id_labels.append(id_label)

    heart_labels = torch.cat(heart_labels)
    id_labels = torch.cat(id_labels)

    # 创建 TensorDataset 和 DataLoader
    denoised_dataset = TensorDataset(denoised_data, heart_labels, id_labels)
    denoised_loader = DataLoader(denoised_dataset, batch_size=original_loader.batch_size, shuffle=False)

    return denoised_loader

# 转换为 DataLoader
filtered_no_id_test_loader = create_denoised_loader(denoised_no_id_signals, no_id_test_loader)

pearson_corr = compute_pearson_correlation(filtered_no_id_test_loader, test_loader)
print(f"Average Pearson Correlation: {pearson_corr}")
rmse = compute_rmse(filtered_no_id_test_loader, test_loader)
print(f"Average RMSE: {rmse}")
frechet_distance = compute_frechet_distance(filtered_no_id_test_loader, test_loader)
print(f"Frechet Distance: {frechet_distance}")

# 在测试集上评估心跳分类模型
test_heart_loss, test_heart_acc = evaluate_heart_model(model_heart, test_loader, heart_criterion, device=device)
print(f'Test Heart Loss: {test_heart_loss:.4f}, Test Heart Accuracy: {test_heart_acc:.4f}')

# 在测试集上评估病人ID分类模型
test_id_loss, test_id_acc = evaluate_id_model(model_id, test_loader, id_criterion, device=device)
print(f'Test ID Loss: {test_id_loss:.4f}, Test ID Accuracy: {test_id_acc:.4f}')


# 在测试集上评估心跳分类模型
test_heart_loss, test_heart_acc = evaluate_heart_model(model_heart, filtered_no_id_test_loader , heart_criterion, device=device)
print(f'Test Heart Loss: {test_heart_loss:.4f}, filtered_no_id_test_loader Test Heart Accuracy: {test_heart_acc:.4f}')

# 在测试集上评估病人ID分类模型
test_id_loss, test_id_acc = evaluate_id_model(model_id, filtered_no_id_test_loader , id_criterion, device=device)
print(f'Test ID Loss: {test_id_loss:.4f},filtered_no_id_test_loader  Test ID Accuracy: {test_id_acc:.4f}')

# 调用函数绘制并保存对比图
# plot_ecg_comparison(filtered_no_id_test_loader, test_loader, num_samples=50, save_path="ecg_comparison_mit_noisefiltered_no_id_test_loader.png")
def plot_ecg_comparison_three(loader1, loader2, loader3, num_samples=50, save_path="ecg_comparison_three.png"):
    """
    绘制三个 DataLoader 的对比图，分别为 test_loader, no_id_test_loader 和 filtered_no_id_test_loader。
    """
    # 随机选择样本的索引
    indices = random.sample(range(len(loader1.dataset)), num_samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 2 * num_samples))

    for i, idx in enumerate(indices):
        # 从三个数据加载器中获取相应的样本
        (original, _, _), (noisy, _, _), (filtered, _, _) = (
            loader1.dataset[idx],
            loader2.dataset[idx],
            loader3.dataset[idx]
        )

        # 确保张量在 CPU 上并转换为 numpy 数组
        original_signal = original.squeeze().cpu().numpy()
        noisy_signal = noisy.squeeze().cpu().numpy()
        filtered_signal = filtered.squeeze().cpu().numpy()

        # 绘制原始信号
        axes[i, 0].plot(original_signal)
        axes[i, 0].set_title(f"Original Signal {idx}")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].set_xlabel("Sample")

        # 绘制带噪信号
        axes[i, 1].plot(noisy_signal)
        axes[i, 1].set_title(f"Noisy Signal {idx}")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].set_xlabel("Sample")

        # 绘制过滤后信号
        axes[i, 2].plot(filtered_signal)
        axes[i, 2].set_title(f"Filtered Signal {idx}")
        axes[i, 2].set_ylabel("Amplitude")
        axes[i, 2].set_xlabel("Sample")

    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)  # 关闭图像以释放内存
    print(f"ECG comparison plot for three loaders saved to {save_path}")


# 调用函数绘制对比图
plot_ecg_comparison_three(
    test_loader,
    no_id_test_loader,
    filtered_no_id_test_loader ,
    num_samples=50,
    save_path="ecg_comparison_three_example_ptb.png"
)

