'''
Homework:
Methodology for solving image classification problems.
Train a simple convolutional neural network (CNN) to classify CIFAR images.
'''

# %%
# importing
import torch
import numpy as np
from matplotlib import pyplot as plt
# %%
# load CIFAR-10 data
from torchvision import datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
train_images, train_labels = train_dataset.data, train_dataset.targets
test_images, test_labels = test_dataset.data, test_dataset.targets
# %%
# Code here!
# inspect images
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(train_images[i])
    ax.set_title(f"Label: {train_labels[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
# %%
# prepocess dataset (include dataloader)
'''
1. one-hot encoded lavels:Transforming the label value into a 10 element binary vector   
(sometimes dont't needed, eg: when using "sparse_categorical_crossentropy" as the loss function in model.compile, we don't need to transfrom the labels, cause this loss function expects integer labels and it does the one-hot encoding it self(sparse_))  
2. Normalize the pixal data, scaling the pixel data from [0,255] to range [0,1]
'''
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
 
# 数据增强和归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 自动将[0,255]转为[0,1]的Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到[-1,1]
])
 
# 创建数据集和数据加载器
batch_size = 64
train_dataset = datasets.CIFAR10(
    root='./data', 
    train=True,
    download=False,
    transform=transform
)
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)
# %%
# create a CNN model
import torch.nn as nn
import torch.nn.functional as F
 
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # 卷积层1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 32x16x16
            
            # 卷积层2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 输出: 64x8x8
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x
 
# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
 
# %%
# train the model
import torch.optim as optim
 
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# 训练参数
num_epochs = 15
 
# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 99:  # 每100个batch打印一次
                print(f'Epoch: {epoch+1}/{num_epochs}, '
                      f'Batch: {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {running_loss/100:.3f}')
                running_loss = 0.0
 
# %%
# evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print(f'Test Accuracy: {100 * correct / total:.2f}%')