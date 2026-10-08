import torch
import torch.nn as nn

class FedPPC_Model(nn.Module):
    """
    专门为 FedPPC 设计的模型：
    1. 显式分离 Encoder 和 Classifier 以进行特征与权重校准
    2. 构造函数支持 num_classes 参数，默认为 2 以适配双输出逻辑
    """
    def __init__(self, input_channels=3, num_classes=2):
        super(FedPPC_Model, self).__init__()
        # Encoder: 提取特征
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 根据输入尺寸计算特征维度 (32x32 -> 64*8*8, 28x28 -> 64*7*7)
        self.feature_dim = 64 * 8 * 8 if input_channels == 3 else 64 * 7 * 7
        
        # Classifier: 权重将与全局原型对齐
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        features = features.reshape(features.size(0), -1)
        logits = self.classifier(features)
        if return_features:
            return features, logits
        return logits