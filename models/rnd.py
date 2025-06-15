"""
Random Network Distillation (RND) 模块
实现随机网络蒸馏的内在动机机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class RNDNetwork(nn.Module):
    """RND网络（目标网络和预测网络共享结构）"""
    
    def __init__(self, input_channels: int = 4, output_dim: int = 512, 
                 conv_layers: list = None):
        super().__init__()
        
        if conv_layers is None:
            conv_layers = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
        
        # 卷积层
        layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size, stride in conv_layers:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.LeakyReLU()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算卷积输出大小
        self.conv_output_size = self._get_conv_output_size()
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def _get_conv_output_size(self):
        """计算卷积层输出大小"""
        x = torch.zeros(1, 4, 84, 84)
        x = self.conv_layers(x)
        return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class RND:
    """Random Network Distillation 类"""
    
    def __init__(self, config, device: torch.device):
        self.device = device
        self.config = config
        
        # 创建目标网络和预测网络
        input_channels = config.get('environment.frame_stack', 4)
        output_dim = config.get('rnd.output_dim', 512)
        conv_layers = config.get('network.conv_layers', [[32, 8, 4], [64, 4, 2], [64, 3, 1]])
        
        # 目标网络（固定权重）
        self.target_network = RNDNetwork(input_channels, output_dim, conv_layers).to(device)
        self.target_network.eval()
        
        # 冻结目标网络参数
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # 预测网络（可训练）
        self.predictor_network = RNDNetwork(input_channels, output_dim, conv_layers).to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.predictor_network.parameters(),
            lr=config.get('rnd.predictor_lr', 0.0001)
        )
        
        # RND参数
        self.intrinsic_reward_coef = config.get('rnd.intrinsic_reward_coef', 1.0)
        self.update_frequency = config.get('rnd.update_frequency', 4)
        self.normalize_intrinsic_reward = config.get('rnd.normalize_intrinsic_reward', True)
        self.intrinsic_reward_clip = config.get('rnd.intrinsic_reward_clip', 5.0)
        
        # 统计信息
        self.update_count = 0
        self.reward_normalizer = RunningMeanStd() if self.normalize_intrinsic_reward else None
        
        # 损失函数
        self.criterion = nn.MSELoss()
    
    def compute_intrinsic_reward(self, states: torch.Tensor) -> torch.Tensor:
        """计算内在奖励"""
        with torch.no_grad():
            # 获取目标网络输出
            target_features = self.target_network(states)
            
            # 获取预测网络输出
            predicted_features = self.predictor_network(states)
            
            # 计算预测误差作为内在奖励
            intrinsic_rewards = F.mse_loss(
                predicted_features, target_features, reduction='none'
            ).mean(dim=1)
            
            # 标准化内在奖励
            if self.normalize_intrinsic_reward and self.reward_normalizer is not None:
                intrinsic_rewards = self.reward_normalizer.normalize(intrinsic_rewards.cpu().numpy())
                intrinsic_rewards = torch.FloatTensor(intrinsic_rewards).to(self.device)
            
            # 裁剪内在奖励
            if self.intrinsic_reward_clip > 0:
                intrinsic_rewards = torch.clamp(
                    intrinsic_rewards, 
                    -self.intrinsic_reward_clip, 
                    self.intrinsic_reward_clip
                )
            
            return intrinsic_rewards * self.intrinsic_reward_coef
    
    def update(self, states: torch.Tensor) -> float:
        """更新预测网络"""
        self.update_count += 1
        
        # 只在指定频率下更新
        if self.update_count % self.update_frequency != 0:
            return 0.0
        
        # 获取目标特征（不计算梯度）
        with torch.no_grad():
            target_features = self.target_network(states)
        
        # 获取预测特征
        predicted_features = self.predictor_network(states)
        
        # 计算损失
        loss = self.criterion(predicted_features, target_features)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.predictor_network.parameters(), 0.5)
        
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'predictor_state_dict': self.predictor_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'reward_normalizer': self.reward_normalizer
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        try:
            # 首先尝试安全加载
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        except Exception:
            # 如果安全加载失败，使用兼容模式
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.predictor_network.load_state_dict(checkpoint['predictor_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.reward_normalizer = checkpoint.get('reward_normalizer', None)


class RunningMeanStd:
    """运行时均值和标准差计算"""
    
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x):
        """更新统计信息"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """从矩更新统计信息"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x):
        """标准化输入"""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class AdaptiveRND(RND):
    """自适应RND，动态调整内在奖励系数"""
    
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)
        
        # 自适应参数
        self.initial_coef = config.get('rnd.intrinsic_reward_coef', 1.0)
        self.min_coef = config.get('rnd.min_intrinsic_coef', 0.01)
        self.decay_rate = config.get('rnd.coef_decay_rate', 0.9999)
        
        # 性能追踪
        self.episode_rewards = []
        self.reward_threshold = config.get('rnd.reward_threshold', 0.0)
        self.adaptation_window = config.get('rnd.adaptation_window', 100)
    
    def adapt_coefficient(self, episode_reward: float):
        """根据性能自适应调整系数"""
        self.episode_rewards.append(episode_reward)
        
        # 保持窗口大小
        if len(self.episode_rewards) > self.adaptation_window:
            self.episode_rewards.pop(0)
        
        # 如果最近的表现良好，减少内在奖励
        if len(self.episode_rewards) >= self.adaptation_window:
            recent_avg = np.mean(self.episode_rewards[-self.adaptation_window//2:])
            overall_avg = np.mean(self.episode_rewards)
            
            if recent_avg > overall_avg and recent_avg > self.reward_threshold:
                self.intrinsic_reward_coef *= self.decay_rate
                self.intrinsic_reward_coef = max(self.intrinsic_reward_coef, self.min_coef)
    
    def compute_intrinsic_reward(self, states: torch.Tensor) -> torch.Tensor:
        """计算内在奖励（带自适应系数）"""
        intrinsic_rewards = super().compute_intrinsic_reward(states)
        return intrinsic_rewards  # 系数已在父类中应用


def create_rnd_module(config, device: torch.device, adaptive: bool = False):
    """创建RND模块的工厂函数"""
    if adaptive:
        return AdaptiveRND(config, device)
    else:
        return RND(config, device)
