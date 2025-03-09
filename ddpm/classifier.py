import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # 卷积层 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入1通道，输出32通道 28 28
        # 卷积层 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入32通道，输出64通道 28 28
        # 最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 池化 14 14
        # Dropout 层 1
        self.dropout1 = nn.Dropout(p=0.25)
        # Flatten 层 (自动展平卷积层输出)
        self.flatten = nn.Flatten() 
        # 全连接层 1
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # 64通道，12x12大小的特征图展平
        # Dropout 层 2
        self.dropout2 = nn.Dropout(p=0.5)
        # 全连接层 2 (输出 10 个类别)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播
        x = self.conv1(x)  # 第一层卷积
        x = F.relu(x)  # ReLU 激活
        x = self.conv2(x)  # 第二层卷积
        x = F.relu(x)  # ReLU 激活
        x = self.maxpool(x)  # 最大池化
        x = self.dropout1(x)  # Dropout
        
        x = self.flatten(x)  # 展平

        x = self.fc1(x)  # 第一个全连接层
        x = torch.relu(x)  # ReLU 激活
        x = self.dropout2(x)  # Dropout

        x = self.fc2(x)  # 输出层
        return x

# 训练模型的函数
    def train_model(self, device, train_loader, val_loader=None, epochs=10, lr=0.001):
        checkpoint_path = 'classifier.pt'
        if os.path.exists(checkpoint_path):
            print("Loading classifier from checkpoint...")
            state = torch.load(checkpoint_path, map_location=device)
            self.load_state_dict(state['model_state_dict'])
            print("Loading classifier completed!")
            return

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(device)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_idx % 100 == 99:  # Print every 100 mini-batches
                    print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0

            if val_loader:
                self.evaluate_model(device, val_loader)

        if not os.path.isdir('results'):
            os.makedirs('results')
        torch.save({
            'model_state_dict': self.state_dict(),
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    # 评估模型的函数
    def evaluate_model(self, device, val_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Accuracy on the test set: {100 * correct / total:.2f}%')