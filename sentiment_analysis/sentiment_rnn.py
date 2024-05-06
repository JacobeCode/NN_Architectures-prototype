import torch
import os
import time

import pandas as pd
import numpy as np

from SentimentAnalysisDataset import SentimentAnalysisDataset
from count_vectorize import count_vectorize
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer

# This data_dir should point for directiory with dataset
data_dir = "D:\\Repos\\Neural_Study\\data\\sentiment_analysis"

# Time calculation start
tstart = time.time()

# Labeling and sorting through data - basic splitting by ";" - and dividing into data and labels
f=open(data_dir + "\\train.txt", "r")
train_data_unclean=f.read()
train_data=train_data_unclean.split("\n")
for num, item in enumerate(train_data):
    train_data[num]=item.split(";")

data=[]
labels=[]
for item in train_data:
    data.append(item[0])
    labels.append(item[1])
    
# Labeling and sorting through data (test_data)
f=open(data_dir + "\\test.txt", "r")
test_data_unclean=f.read()
test_data=test_data_unclean.split("\n")
for num, item in enumerate(test_data):
    test_data[num]=item.split(";")

test_dataset=[]
test_labels=[]
for item in test_data:
    test_dataset.append(item[0])
    test_labels.append(item[1])

# Converting simple arrays to DataFrame [for visualization purpose - train + test]
train_database=pd.DataFrame(data=data)
train_database["sentiment"]=labels
train_database.columns=["text_data", "sentiment"]

# Vectorization with help of sklearn - fit_transform of train data and transform of test data
count_vector = CountVectorizer()
data = count_vector.fit_transform(data)
test_dataset = count_vector.transform(test_dataset)

# Converting labels to numeric values - One Hot Encoding
unique_labels = np.unique(labels)
numeric_labels = []
for iter, item in enumerate(labels):
    for labs in unique_labels:
        if item == labs:
            numeric_labels.append(np.where(unique_labels == labs)[0][0])
        else:
            pass
        
unique_labels = np.unique(labels)
numeric_test_labels = []
for iter, item in enumerate(test_labels):
    for labs in unique_labels:
        if item == labs:
            numeric_test_labels.append(np.where(unique_labels == labs)[0][0])
        else:
            pass

# Converting vectroized data to proper format
X_train = torch.from_numpy(data.todense()).float()
X_test = torch.from_numpy(test_dataset.todense()).float()

# Initialiazing the train\test datasets
train_dataset = SentimentAnalysisDataset(text_data=X_train, text_labels=numeric_labels)
test_dataset = SentimentAnalysisDataset(text_data=X_test, text_labels=numeric_test_labels)

# Init for DataLoader objects (default batch = 64)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, drop_last=True)

# Checking for CUDA avalibility
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using " + str(device))

# Defining NN architecture
class SentimentNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn1 = nn.RNN(input_size=15186, hidden_size=10, num_layers=2, batch_first=True, bidirectional=False)

        self.lin1 = nn.Linear(15186, 7500)
        self.act1 = nn.ReLU()

        self.lin2 = nn.Linear(7500, 10)
        self.soft = nn.Softmax()

    def forward(self, x):
        hidden = None

        # x = self.act1(self.lin1(x))
        x, hidden = self.rnn1(x, hidden)
        x = self.soft(x)
        return x

# Model initialization (object)    
model = SentimentNetwork().to(device)
print(model)

# Loss_fn and optimizer (default - CCE and SGD on 1e-3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Train mode for obj
model.train()

# Train and test functions - classic flow of training with backpropagation and calculating basic measures (with saving to file)
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
            print(f"loss: {loss:>7f}")              # This prints current tensors : [{current:>5d}/{size:>5d}]
            loss_stack.append(loss)

    pd.DataFrame(loss_stack).to_csv("sentiment_analysis\\training_data\\loss_" + str(epoche))

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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return 100*correct, test_loss

avg_loss_stack = []
acc_stack = []
test_stack = []

# Loop for training process for x epochs
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_loss = train(train_dataloader, model, loss_fn, optimizer, t)
    avg_loss_stack.append(avg_loss)
    acc, test_loss = test(test_dataloader, model, loss_fn)
    acc_stack.append(acc)
    test_stack.append(test_loss)
print("Done!")

# Saving accuracy and avg loss to .csv files
pd.DataFrame(acc_stack).to_csv("sentiment_analysis\\training_data\\acc_epoche")
pd.DataFrame(test_stack).to_csv("sentiment_analysis\\training_data\\avg_loss_epoche")

# Summation of process
print("Elapsed time: " + str(time.time() - tstart))
print("Accuracy: " + str(acc_stack))
print("Avg loss: " + str(test_stack))

# Saving rest of train data
with open ("sentiment_analysis\\training_data\\model_info", "w") as f:
    f.write(f"Elapsed time: {time.time() - tstart}\n\n")
    f.write(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        f.write(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")

f.close()