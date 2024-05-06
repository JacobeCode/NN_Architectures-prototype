# NOTES:
# - adding LSTM to neural net
# - remake part of data processing and moving it whole to dataloader
# - adding dynamic batch padding in stead of padding whole dataset
# - add automatic calculation of input in the first Linear layer
# - add randomizing samples from whole dataset with np.random.choice()

# Used dataset: https://github.com/soerenab/AudioMNIST

import torch
import os
import time

from torch.utils.data import DataLoader, Dataset
from DigitRecognitionDataset import DigitRecognitionDataset

import torch.nn as nn
import librosa as lb
import numpy as np
import pandas as pd
import random as rd

# IMPORTANT - take into consideration that processing of audio data can take some time if done manually through this code
# App. for GTX1650 ~ 25 min for 5500 subset

# Parameters for processing
fs = 44100        # Sampling frequency for audio (loading and MFCC)
num_batch = 64   # Batch Size for DataLoader and Model

# Recordings load with padding to longest audio (with added normalization) [in first 500 is longest so it is best for testing setup]
data_dir = 'D:\\Repos\\Neural_Study\\data\\digit_recognition'
train_wav_list = os.listdir(data_dir+ "\\train")
test_wav_list = os.listdir(data_dir + "\\test")
train_recordings = []
test_recordings = []
train_audio_labels = []
test_audio_labels = []

# Run start
tstart = time.time()

# Manual data processing - loading audio and auto-padding to the longest audio in the dataset (test set is adjusted to train length)
for iter, rec in enumerate(train_wav_list):
    temp = lb.load(data_dir + "\\train" + '\\' + rec, sr=fs)[0]
    train_audio_labels.append(str(train_wav_list[iter][0]))
    print(str(iter+1) + "\\" + str(len(train_wav_list)))
    if iter != 0: 
        data_len = np.shape(train_recordings)[-1]
        if len(temp) < data_len:
            temp = np.append(temp, np.zeros(np.abs(data_len-len(temp))))
        elif len(temp) > data_len:
            if np.ndim(train_recordings) == 1:
                train_recordings = np.hstack((train_recordings, np.zeros((np.abs(data_len-len(temp))))))
            else:
                train_recordings = np.pad(train_recordings, pad_width=((0, 0), (0, np.abs(data_len-len(temp)))), constant_values=0)
        else: pass
        train_recordings = np.vstack((train_recordings, temp/max(np.abs(temp))))
    else: train_recordings = temp/max(np.abs(temp))

for iter, rec in enumerate(test_wav_list):
    temp = lb.load(data_dir + "\\test" + '\\' + rec, sr=fs)[0]
    test_audio_labels.append(test_wav_list[iter][0])
    data_len = np.shape(train_recordings)[-1]
    print(str(iter+1) + "\\" + str(len(test_wav_list)))
    if len(temp) < data_len:
        temp = np.append(temp, np.zeros(np.abs(data_len-len(temp))))
    elif len(temp) > data_len: 
        temp = temp[0:data_len]
    else: pass
    if iter !=0:
        test_recordings = np.vstack((test_recordings, temp/max(np.abs(temp))))
    else: test_recordings = temp/max(np.abs(temp))

train_audio_labels = np.array(train_audio_labels)
test_audio_labels = np.array(test_audio_labels)

# This one can save some recording data to .csv (I don't recommend it with a lot of audio - 500 items is around 1,5GB)
# pd.DataFrame(recordings).to_csv("rec_data")

# Basic stripped MFCC calculating - in later versions it can be improved, but now I want only to experiment with PyTorch
train_MFCC_list = []
test_MFCC_list = []
for iter, record in enumerate(train_recordings):
    print(str(iter+1) + "\\" + str(len(train_recordings)))
    if iter != 0:       
        train_MFCC_list = np.append(train_MFCC_list, np.expand_dims(lb.feature.mfcc(y=record, sr=fs), (0, 1)), axis=0)
    else:
        train_MFCC_list = np.expand_dims(lb.feature.mfcc(y=record, sr=fs), (0, 1))

for iter, record in enumerate(test_recordings):
    print(str(iter+1) + "\\" + str(len(test_recordings)))
    if iter != 0:       
        test_MFCC_list = np.append(test_MFCC_list, np.expand_dims(lb.feature.mfcc(y=record, sr=fs), (0, 1)), axis=0)
    else:
        test_MFCC_list = np.expand_dims(lb.feature.mfcc(y=record, sr=fs), (0, 1))

print(np.shape(test_MFCC_list))

# Example of data extraction
# print(np.shape(MFCC_list[:,:,0]))

# Saving MFCC to .csv file
# pd.DataFrame(train_MFCC_list).to_csv("MFCC_train_data")
# pd.DataFrame(test_MFCC_list).to_csv("MFCC_test_data")

# PyTorch train instance - here for example could be shown examplatory MFCC
train_dataset = DigitRecognitionDataset(audio_labels=train_audio_labels, audio_data=train_MFCC_list)
test_dataset = DigitRecognitionDataset(audio_labels=test_audio_labels, audio_data=test_MFCC_list)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=num_batch, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=num_batch, shuffle=True, drop_last=True)

# Checking CUDA avalibility
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using " + str(device))

# Neural Network class definition with forward pass definiton
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,4))
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2,2), stride=2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.batch = nn.BatchNorm2d(num_features=16)
        self.flat = nn.Flatten(start_dim=1, end_dim=3)

        self.lstm1 = nn.LSTM(input_size=320, hidden_size=256, num_layers=10, bidirectional=True, batch_first=True) # input = 320 for 5500
        self.actlstm = nn.ReLU()

        self.lin1 = nn.Linear(512, 256)
        self.act3 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.batch1 = nn.BatchNorm1d(num_features=256)

        self.lin2 = nn.Linear(256, 144)
        self.act4 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        self.batch2 = nn.BatchNorm1d(num_features=144)

        self.lin3 = nn.Linear(144, 64)
        self.act5 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.batch3 = nn.BatchNorm1d(num_features=64)

        self.lin4 = nn.Linear(64, 10)
        self.act6 = nn.Softmax()

    def forward(self, x):
        hidden = None

        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.batch(x)
        x = self.flat(x)
        x, hidden = self.lstm1(x, hidden)
        x = self.actlstm(x)
        x = self.act3(self.lin1(x))
        x = self.batch1(x)
        x = self.act4(self.lin2(x))
        x = self.batch2(x)
        x = self.act5(self.lin3(x))
        x = self.batch3(x)
        x = self.act6(self.lin4(x))
        return x

model = NeuralNetwork().to(device)
print(model)

# Loss_fn and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)

model.train()

# Train and test loop for training process - classic with backpropagation
def train(train_dataloader, model, loss_fn, optimizer, epoche):
    
    loss_stack = []

    size = len(train_dataloader.dataset)                  # size for parameters calculation
    model.train()

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()                            # next step in optimizer algorithm
        optimizer.zero_grad()                       # "zeroing" the gradients of optimized torch.Tensor's - set_to_none param - changing to None = performance better
                                                    # but it can alternate some of behaviour e.g.'s in documentation

        # printing relevant data - loss function and current bath and size
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            loss_stack.append(loss)

    pd.DataFrame(loss_stack).to_csv("digit_recognition\\training_data\\loss_" + str(epoche))

    return np.average(loss_stack)

def test(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():                       # disabling grad calculation for efficiency
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()    # .item() returning single value tensor as Python number
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()     # checking classes of pred by argmax and comparing them to labels
    
    # Set up for avarages 
    test_loss /= num_batches
    correct /= size
    # size = len(train_dataloader.dataset)

    return 100*correct, test_loss

avg_loss_stack = []
acc_stack = []
test_stack = []

# Training the network
epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_loss = train(train_dataloader, model, loss_fn, optimizer, t)
    avg_loss_stack.append(avg_loss)
    acc, test_loss = test(test_dataloader, model, loss_fn)
    acc_stack.append(acc)
    test_stack.append(test_loss)
print("Done!")

# Saving train data to .csv file
pd.DataFrame(acc_stack).to_csv("digit_recognition\\training_data\\acc_epoche")
pd.DataFrame(test_stack).to_csv("digit_recognition\\training_data\\avg_loss_epoche")

# Summation of training
print("Elapsed time: " + str(time.time() - tstart))
print("Accuracy: " + str(acc_stack))
print("Avg loss: " + str(test_stack))

# Saving rest of parameters and model details
with open ("digit_recognition\\training_data\\model_info", "w") as f:
    f.write(f"Elapsed time: {time.time() - tstart}\n\n")
    f.write(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        f.write(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")

f.close()