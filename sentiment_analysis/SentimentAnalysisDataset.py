# Dataset for PyTorch sentiment analysis

from torch.utils.data import DataLoader, Dataset

import numpy as np

class SentimentAnalysisDataset(Dataset):
    def __init__(self, text_data, text_labels, transform = None, target_transform = None):
        super().__init__()
        self.text_data = text_data
        self.text_labels = text_labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.text_labels)

    def __getitem__(self, index):
        sample = self.text_data[index]
        label = self.text_labels[index]

        return sample, label