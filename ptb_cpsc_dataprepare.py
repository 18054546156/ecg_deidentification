import os
import numpy as np
import wfdb
import h5py
import torch
from torch.utils.data import Dataset
from scipy.signal import resample
from sklearn.model_selection import train_test_split

# 下载后的 PTB-XL 和 CPSC 数据集目录
PTBXL_DATA_DIR = 'path_to_ptbxl'
CPSC_DATA_DIR = 'path_to_cpsc2018'


# Function to downsample ECG waves
def downsample_waves(waves, new_size):
    return np.array([resample(wave, new_size, axis=1) for wave in waves])


# Function to remove invalid samples (NaN values, or first 15 timesteps being all zeros)
def remove_invalid_samples(waves):
    nan_mask = np.isnan(waves).any(axis=(1, 2))
    zero_mask = (np.abs(waves[:, :, :15]).sum(axis=(1, 2)) == 0)
    valid_indices = ~(nan_mask | zero_mask)
    print(f'Invalid samples: {np.sum(~valid_indices)}')
    return waves[valid_indices]


# Custom Dataset class for ECG data
class ECGDataset(Dataset):
    def __init__(self, waves, labels=None, transform=None):
        self.waves = torch.tensor(waves, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None
        self.transform = transform

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        wave = self.waves[idx]
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        if self.transform:
            wave = self.transform(wave)
        return wave, label


# Function to extract diagnosis codes from WFDB records (for CPSC2018)
def extract_diagnosis_code(record):
    for comment in record.comments:
        if comment.startswith('Dx:'):
            return comment.split(': ')[1]
    return None


# Function to read and preprocess PTB-XL data
def load_ptbxl_data(data_dir, reduced_lead=True, downsample=True):
    import pandas as pd
    from scipy.io import loadmat

    # Load the CSV file with labels
    labels_df = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'))
    data, labels = [], []

    # Load the .mat files for each record
    for _, row in labels_df.iterrows():
        mat_file = os.path.join(data_dir, 'records500', row['filename_hr'])
        mat_data = loadmat(mat_file)['val']

        # Transpose to have (n_channels, n_timesteps)
        mat_data = mat_data.T
        data.append(mat_data)
        labels.append(row['scp_codes'])  # Assuming 'scp_codes' contains the labels

    # Stack data and convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Apply lead reduction (e.g., keep only 8 specific leads)
    if reduced_lead:
        data = np.concatenate([data[:, :2], data[:, 6:]], axis=1)

    # Apply downsampling (e.g., to 2500 samples)
    if downsample:
        data = downsample_waves(data, 2500)

    return data, labels


# Function to read and preprocess CPSC2018 data
def load_cpsc_data(data_dir, reduced_lead=True, downsample=True):
    waves, labels = [], []

    for filename in os.listdir(data_dir):
        if filename.endswith('.hea'):
            record_name = os.path.splitext(filename)[0]
            record = wfdb.rdrecord(os.path.join(data_dir, record_name))
            ecg_data = record.p_signal
            ecg_label = extract_diagnosis_code(record)

            # Resample the data if necessary
            if record.fs != 500:
                new_length = int((500 / record.fs) * record.sig_len)
                ecg_data = resample(ecg_data, new_length)

            waves.append(ecg_data)
            labels.append(ecg_label)

    # Stack the data
    waves = np.stack(waves)
    labels = np.array(labels)

    # Apply lead reduction
    if reduced_lead:
        waves = np.concatenate([waves[:, :2], waves[:, 6:]], axis=1)

    # Apply downsampling
    if downsample:
        waves = downsample_waves(waves, 2500)

    return remove_invalid_samples(waves), labels


# Create PyTorch datasets and dataloaders
def create_datasets(ptbxl_data_dir, cpsc_data_dir, batch_size=32, test_size=0.2):
    # Load PTB-XL and CPSC2018 data
    ptbxl_waves, ptbxl_labels = load_ptbxl_data(ptbxl_data_dir)
    cpsc_waves, cpsc_labels = load_cpsc_data(cpsc_data_dir)

    # Split PTB-XL and CPSC data into train/test sets
    ptbxl_train_waves, ptbxl_test_waves, ptbxl_train_labels, ptbxl_test_labels = train_test_split(ptbxl_waves,
                                                                                                  ptbxl_labels,
                                                                                                  test_size=test_size)
    cpsc_train_waves, cpsc_test_waves, cpsc_train_labels, cpsc_test_labels = train_test_split(cpsc_waves, cpsc_labels,
                                                                                              test_size=test_size)

    # Create PyTorch datasets
    ptbxl_train_dataset = ECGDataset(ptbxl_train_waves, ptbxl_train_labels)
    ptbxl_test_dataset = ECGDataset(ptbxl_test_waves, ptbxl_test_labels)

    cpsc_train_dataset = ECGDataset(cpsc_train_waves, cpsc_train_labels)
    cpsc_test_dataset = ECGDataset(cpsc_test_waves, cpsc_test_labels)

    # Create DataLoaders
    ptbxl_train_loader = torch.utils.data.DataLoader(ptbxl_train_dataset, batch_size=batch_size, shuffle=True)
    ptbxl_test_loader = torch.utils.data.DataLoader(ptbxl_test_dataset, batch_size=batch_size, shuffle=False)

    cpsc_train_loader = torch.utils.data.DataLoader(cpsc_train_dataset, batch_size=batch_size, shuffle=True)
    cpsc_test_loader = torch.utils.data.DataLoader(cpsc_test_dataset, batch_size=batch_size, shuffle=False)

    return ptbxl_train_loader, ptbxl_test_loader, cpsc_train_loader, cpsc_test_loader


# Example usage:
if __name__ == '__main__':
    ptbxl_train_loader, ptbxl_test_loader, cpsc_train_loader, cpsc_test_loader = create_datasets(PTBXL_DATA_DIR,
                                                                                                 CPSC_DATA_DIR)
    print(f'PTB-XL train samples: {len(ptbxl_train_loader.dataset)}')
    print(f'CPSC train samples: {len(cpsc_train_loader.dataset)}')
