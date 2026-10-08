import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 基础模块定义 (用于 ResNet)
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ==========================================
# ResNet 核心类
# ==========================================
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10, input_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels)

# ==========================================
# 自动适配维度的 SimpleCNN (核心修改点)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, prior=0.5, input_channels=3):
        super(SimpleCNN, self).__init__()
        self.prior = prior
        self.input_channels = input_channels
        
        # 卷积部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 动态计算 Flatten 后的维度
        self.feature_dim = self._get_flatten_dim()
        
        # 全连接层
        self.fc = nn.Linear(self.feature_dim, 1)

    def _get_flatten_dim(self):
        """通过模拟一次前向传播自动计算特征维度"""
        with torch.no_grad():
            # 适配不同数据集的分辨率
            # MNIST/F-MNIST 通常是 28, CIFAR-10 是 32
            size = 28 if self.input_channels == 1 else 32
            dummy_input = torch.zeros(1, self.input_channels, size, size)
            dummy_output = self.conv_layers(dummy_input)
            return dummy_output.numel()

    def forward(self, x):
        h = self.conv_layers(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)

# ==========================================
# CNNMnist (适配 PU 学习)
# ==========================================
class CNNMnist(nn.Module):
    def __init__(self, num_classes=1):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # 对于 28x28 输入，经过两次 5x5 卷积和池化后展平为 320
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# ==========================================
# 其他模型
# ==========================================
class ThreeLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(ThreeLayerPerceptron, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)