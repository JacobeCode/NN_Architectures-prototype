import os
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class DigitRecognitionDataset(Dataset):
    def __init__(self, audio_labels, audio_data, transform=None, target_transform=None):
        self.audio_data=audio_data
        self.audio_labels=np.array(audio_labels)
        self.transform=transform
        self.target_transform=target_transform

    def __len__(self):
        return len(self.audio_labels)
    
    def __getitem__(self, idx):
        data = self.audio_data[idx,:,:,:].astype(np.float32)
        label = self.audio_labels[idx].astype(np.int64)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label