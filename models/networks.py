"""
神经网络模型模块
实现DQN、Dueling DQN等网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class CNNFeatureExtractor(nn.Module):
    """卷积特征提取器"""
    
    def __init__(self, input_channels: int = 4, conv_layers: list = None):
        super().__init__()
        
        if conv_layers is None:
            conv_layers = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
        
        layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size, stride in conv_layers:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.ReLU()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算输出特征大小
        self.feature_size = self._get_conv_output_size()
    
    def _get_conv_output_size(self):
        """计算卷积层输出大小"""
        x = torch.zeros(1, 4, 84, 84)  # 假设输入为84x84x4
        x = self.conv_layers(x)
        return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1)


class DQN(nn.Module):
    """标准DQN网络"""
    
    def __init__(self, input_channels: int = 4, num_actions: int = 18, 
                 hidden_dim: int = 512, conv_layers: list = None):
        super().__init__()
        
        self.num_actions = num_actions
        
        # 特征提取器
        self.feature_extractor = CNNFeatureExtractor(input_channels, conv_layers)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        q_values = self.fc(features)
        return q_values
    
    def get_features(self, x):
        """获取特征表示"""
        return self.feature_extractor(x)


class DuelingDQN(nn.Module):
    """Dueling DQN网络"""
    
    def __init__(self, input_channels: int = 4, num_actions: int = 18, 
                 hidden_dim: int = 512, conv_layers: list = None):
        super().__init__()
        
        self.num_actions = num_actions
        
        # 特征提取器
        self.feature_extractor = CNNFeatureExtractor(input_channels, conv_layers)
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 组合价值和优势
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def get_features(self, x):
        """获取特征表示"""
        return self.feature_extractor(x)


class NoisyLinear(nn.Module):
    """噪声线性层，用于NoisyNet"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 权重参数
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        # 偏置参数
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int):
        """生成噪声"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class NoisyDQN(nn.Module):
    """Noisy DQN网络"""
    
    def __init__(self, input_channels: int = 4, num_actions: int = 18, 
                 hidden_dim: int = 512, conv_layers: list = None, std_init: float = 0.5):
        super().__init__()
        
        self.num_actions = num_actions
        
        # 特征提取器
        self.feature_extractor = CNNFeatureExtractor(input_channels, conv_layers)
        
        # 噪声全连接层
        self.noisy_fc1 = NoisyLinear(self.feature_extractor.feature_size, hidden_dim, std_init)
        self.noisy_fc2 = NoisyLinear(hidden_dim, num_actions, std_init)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        x = F.relu(self.noisy_fc1(features))
        q_values = self.noisy_fc2(x)
        return q_values
    
    def reset_noise(self):
        """重置噪声"""
        self.noisy_fc1.reset_noise()
        self.noisy_fc2.reset_noise()
    
    def get_features(self, x):
        """获取特征表示"""
        return self.feature_extractor(x)


class RainbowDQN(nn.Module):
    """Rainbow DQN网络（结合Dueling和Noisy）"""
    
    def __init__(self, input_channels: int = 4, num_actions: int = 18, 
                 hidden_dim: int = 512, conv_layers: list = None, 
                 num_atoms: int = 51, v_min: float = -10, v_max: float = 10,
                 std_init: float = 0.5):
        super().__init__()
        
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # 特征提取器
        self.feature_extractor = CNNFeatureExtractor(input_channels, conv_layers)
        
        # 价值流（噪声）
        self.value_fc1 = NoisyLinear(self.feature_extractor.feature_size, hidden_dim, std_init)
        self.value_fc2 = NoisyLinear(hidden_dim, num_atoms, std_init)
        
        # 优势流（噪声）
        self.advantage_fc1 = NoisyLinear(self.feature_extractor.feature_size, hidden_dim, std_init)
        self.advantage_fc2 = NoisyLinear(hidden_dim, num_actions * num_atoms, std_init)
    
    def forward(self, x):
        batch_size = x.size(0)
        features = self.feature_extractor(x)
        
        # 价值流
        value = F.relu(self.value_fc1(features))
        value = self.value_fc2(value).view(batch_size, 1, self.num_atoms)
        
        # 优势流
        advantage = F.relu(self.advantage_fc1(features))
        advantage = self.advantage_fc2(advantage).view(batch_size, self.num_actions, self.num_atoms)
        
        # 组合价值和优势
        q_atoms = value + advantage - advantage.mean(1, keepdim=True)
        
        # 应用softmax得到分布
        q_dist = F.softmax(q_atoms, dim=2)
        
        return q_dist
    
    def reset_noise(self):
        """重置噪声"""
        self.value_fc1.reset_noise()
        self.value_fc2.reset_noise()
        self.advantage_fc1.reset_noise()
        self.advantage_fc2.reset_noise()
    
    def get_features(self, x):
        """获取特征表示"""
        return self.feature_extractor(x)


def create_dqn_network(config, num_actions: int = 18, network_type: str = "dueling"):
    """创建DQN网络的工厂函数"""
    
    input_channels = config.get('environment.frame_stack', 4)
    hidden_dim = config.get('network.hidden_dim', 512)
    conv_layers = config.get('network.conv_layers', [[32, 8, 4], [64, 4, 2], [64, 3, 1]])
    
    if network_type == "standard":
        return DQN(input_channels, num_actions, hidden_dim, conv_layers)
    elif network_type == "dueling":
        return DuelingDQN(input_channels, num_actions, hidden_dim, conv_layers)
    elif network_type == "noisy":
        std_init = config.get('network.noisy_std', 0.5)
        return NoisyDQN(input_channels, num_actions, hidden_dim, conv_layers, std_init)
    elif network_type == "rainbow":
        num_atoms = config.get('network.num_atoms', 51)
        v_min = config.get('network.v_min', -10)
        v_max = config.get('network.v_max', 10)
        std_init = config.get('network.noisy_std', 0.5)
        return RainbowDQN(input_channels, num_actions, hidden_dim, conv_layers, 
                         num_atoms, v_min, v_max, std_init)
    else:
        raise ValueError(f"Unsupported network type: {network_type}")


def init_weights(module):
    """初始化网络权重"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
