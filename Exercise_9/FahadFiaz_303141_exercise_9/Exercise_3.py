
from sklearn import datasets
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
import time


############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs"
writer = SummaryWriter()
###################################################

#load datset
olivetti = datasets.fetch_olivetti_faces()

# Hyper-parameters 
input_size = 4096 # 28x28 
num_classes = 40
num_epochs = 100
batch_size = 10
learning_rate = 0.01

class OlivettifacesDataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        self.n_samples = olivetti.images.shape[0]

        self.x_data = torch.from_numpy(olivetti.images) # size [n_samples, n_features]
        self.x_data = torch.unsqueeze(self.x_data, 1)
        self.y_data = torch.from_numpy(olivetti.target) # size [n_samples, 1]
        self.y_data= self.y_data.type(torch.LongTensor)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset
dataset = OlivettifacesDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False)

#convert to an iterator and look at one random sample
examples = iter(test_loader)
example_data, example_targets = examples.next()

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5) #input channel size, output channel size ,kernal size
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 30 * 30, 120)
        self.fc2 = nn.Linear(120, 40)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 10 * 30 * 30)           
        x = F.relu(self.fc1(x))             
        x = self.fc2(x)               
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = NeuralNet()


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

############## TENSORBOARD ########################
writer.add_graph(model, example_data)
###################################################

# Train the model
running_loss = 0.0
n_total_steps = len(train_loader)

start_time = time.time()

for epoch in range(num_epochs):
    n_correct = 0.0
    n_samples = 0.0
    for i, (images, labels) in enumerate(train_loader):  
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        
    if (epoch+1)% 1 == 0:
          writer.add_histogram('ConvLayer.bias', model.conv1.bias, epoch+1)
          writer.add_histogram('ConvLayer.weight', model.conv1.weight, epoch+1) 
          writer.add_histogram('HiddenLayer.bias', model.fc2.bias, epoch+1)
          writer.add_histogram('HiddenLayer.weight', model.fc2.weight, epoch+1)

    if (epoch+1)% 10 == 0:
        acc = 100.0 * n_correct / n_samples
        print(f'Epoch [{epoch+1}/{num_epochs}], Running loss {int(running_loss)},Accuracy of the network on the {n_samples} training images: {acc} %')
        
    ############## TENSORBOARD ########################
        writer.add_scalar('training loss', running_loss, epoch+1)
        writer.add_scalar('training accuracy', acc, epoch+1)  
        running_loss=0.0      
    ###################################################

print(f'Total Epochs: 100, Training time taken: {time.time() - start_time:.2f} seconds')
print(f'Number of traninable paramters: {count_parameters(model)}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    running_correct = 0.0
    running_sample = 0.0

    for i,(images, labels) in enumerate(test_loader):
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        running_sample += labels.size(0)
        running_correct += (predicted == labels).sum().item()
        running_loss += loss.item()

        if (i+1) % 1 == 0:
          acc = 100.0 * running_correct / running_sample
          ############## TENSORBOARD ########################
          writer.add_scalar('Prediction accuracy', acc, i+1)  
          running_correct = 0.0
          running_sample = 0.0
          ###################################################


    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')
writer.close()

# %load_ext tensorboard
# %tensorboard --logdir runs/







