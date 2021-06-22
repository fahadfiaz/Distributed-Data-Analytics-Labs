
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs"
writer = SummaryWriter()
###################################################

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size =500
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.01

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()

class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5) #input channel size, output channel size ,kernal size
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(10 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 10)
        
        self.fr1 = nn.Linear(10 * 12 * 12, 125)
        self.fr2 = nn.Linear(125, 100)

    def forward(self, input):
        
        x = self.pool(F.relu(self.conv1(input)))
        x = x.view(-1, 10 * 12 * 12)  
        
        fc_classifier = F.relu(self.fc1(x))             
        classifier_out = self.fc2(fc_classifier)
        
        
        fc_regression = F.relu(self.fr1(x))
        regression_out = self.fr2(fc_regression)
        
        outputs = [classifier_out, regression_out]
        return outputs

class MultiTaskLoss(nn.Module):
    def __init__(self, model, loss_fn):
        super(MultiTaskLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        
    def forward(self, input, targets):
        outputs = self.model(input)
        
        classification_loss = self.loss_fn[0](outputs[0],targets[0])
        regression_loss = self.loss_fn[1](outputs[1],targets[1])
        total_loss = classification_loss + regression_loss
        classification_prediction = outputs[0]
        
        return classification_loss,regression_loss, total_loss,classification_prediction

loss_fn1 = nn.CrossEntropyLoss()
loss_fn2 = nn.L1Loss()
model=MultiTaskModel()
mtl = MultiTaskLoss(model=model.to(device),loss_fn=[loss_fn1, loss_fn2])
optimizer = torch.optim.Adam(mtl.parameters(), lr=learning_rate)

############## TENSORBOARD ########################
writer.add_graph(MultiTaskModel(), example_data)
###################################################

# Train the model

running_loss = 0.0
running_classification_loss = 0.0
running_regression_loss = 0.0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        
        # Forward pass

        images = images.to(device) #shifting data to gpu if available
        labels = labels.to(device)
        labels2 = labels.to(device)

        loss_cl,loss_reg, total_loss, _ = mtl(images,[labels, labels])
        
        running_loss += total_loss.item()    
        running_classification_loss += loss_cl.item()
        running_regression_loss += loss_reg.item()
      
    if (epoch+1)% 1 == 0:
          writer.add_histogram('ConvLayer.bias', model.conv1.bias, epoch+1)
          writer.add_histogram('ConvLayer.weight', model.conv1.weight, epoch+1) 

          writer.add_histogram('ClassificationLayer.bias', model.fc2.bias, epoch+1)
          writer.add_histogram('ClassificationLayer.weight', model.fc2.weight, epoch+1)

          writer.add_histogram('RegressionLayer.bias', model.fr2.bias, epoch+1)
          writer.add_histogram('RegressionLayer.weight', model.fr2.weight, epoch+1)
        
    if (epoch+1)% 1 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Total_loss:{running_loss:.2f}, Classification_loss:{running_classification_loss:.2f}, Regression_loss:{running_regression_loss:.2f}")
        ############## TENSORBOARD ########################
        writer.add_scalar('Prediction total loss', running_loss, i+1)  
        writer.add_scalar('Prediction classification loss', running_classification_loss, i+1)
        writer.add_scalar('Prediction regression loss', running_regression_loss, i+1)  
        ###################################################

        running_loss = 0.0
        running_classification_loss = 0.0
        running_regression_loss = 0.0

#Testing model

# In test phase, we don't need to compute gradients (for memory efficiency)
class_labels = []
class_preds = []
acc_tl=[]
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i,(images, labels) in enumerate(test_loader):
    
        outputs = mtl(images,[labels, labels])
        
        _ ,_ , _ , predictions = mtl(images,[labels, labels])
        
        # max returns (value ,index)
        values, predicted = torch.max(predictions.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the classification network on the 10000 test images: {acc} %')
