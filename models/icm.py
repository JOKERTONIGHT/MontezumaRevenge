"""
Intrinsic Curiosity Module (ICM) 模块
实现基于好奇心的内在动机机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class FeatureNetwork(nn.Module):
    """特征提取网络"""
    
    def __init__(self, input_channels: int = 4, feature_dim: int = 288, conv_layers: list = None):
        super().__init__()
        
        if conv_layers is None:
            conv_layers = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
        
        # 卷积层
        layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size, stride in conv_layers:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.ELU()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算卷积输出大小
        self.conv_output_size = self._get_conv_output_size()
        
        # 特征映射层
        self.feature_fc = nn.Linear(self.conv_output_size, feature_dim)
    
    def _get_conv_output_size(self):
        """计算卷积层输出大小"""
        x = torch.zeros(1, 4, 84, 84)
        x = self.conv_layers(x)
        return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.feature_fc(x)
        return features


class InverseModel(nn.Module):
    """逆向模型：预测动作 a_t = f^{-1}(s_t, s_{t+1})"""
    
    def __init__(self, feature_dim: int = 288, num_actions: int = 18, hidden_dim: int = 256):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, state_features, next_state_features):
        """预测从state到next_state的动作"""
        combined_features = torch.cat([state_features, next_state_features], dim=1)
        action_logits = self.fc(combined_features)
        return action_logits


class ForwardModel(nn.Module):
    """前向模型：预测下一状态特征 φ(s_{t+1}) = f(φ(s_t), a_t)"""
    
    def __init__(self, feature_dim: int = 288, num_actions: int = 18, hidden_dim: int = 256):
        super().__init__()
        
        self.action_embedding = nn.Embedding(num_actions, feature_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, state_features, actions):
        """预测下一状态特征"""
        action_embeddings = self.action_embedding(actions)
        combined_features = torch.cat([state_features, action_embeddings], dim=1)
        predicted_next_features = self.fc(combined_features)
        return predicted_next_features


class ICM:
    """Intrinsic Curiosity Module"""
    
    def __init__(self, config, device: torch.device, num_actions: int = 18):
        self.device = device
        self.config = config
        self.num_actions = num_actions
        
        # 网络参数
        input_channels = config.get('environment.frame_stack', 4)
        feature_dim = config.get('icm.feature_dim', 288)
        conv_layers = config.get('network.conv_layers', [[32, 8, 4], [64, 4, 2], [64, 3, 1]])
        hidden_dim = config.get('icm.hidden_dim', 256)
        
        # 创建网络
        self.feature_network = FeatureNetwork(input_channels, feature_dim, conv_layers).to(device)
        self.inverse_model = InverseModel(feature_dim, num_actions, hidden_dim).to(device)
        self.forward_model = ForwardModel(feature_dim, num_actions, hidden_dim).to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.feature_network.parameters()) + 
            list(self.inverse_model.parameters()) + 
            list(self.forward_model.parameters()),
            lr=config.get('icm.learning_rate', 0.001)
        )
        
        # ICM参数
        self.intrinsic_reward_coef = config.get('icm.intrinsic_reward_coef', 0.01)
        self.forward_loss_coef = config.get('icm.forward_loss_coef', 0.2)
        self.inverse_loss_coef = config.get('icm.inverse_loss_coef', 0.8)
        self.update_frequency = config.get('icm.update_frequency', 4)
        
        # 统计信息
        self.update_count = 0
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def compute_intrinsic_reward(self, states: torch.Tensor, actions: torch.Tensor, 
                                next_states: torch.Tensor) -> torch.Tensor:
        """计算内在奖励"""
        with torch.no_grad():
            # 提取特征
            state_features = self.feature_network(states)
            next_state_features = self.feature_network(next_states)
            
            # 前向模型预测
            predicted_next_features = self.forward_model(state_features, actions)
            
            # 计算预测误差作为内在奖励
            prediction_error = F.mse_loss(
                predicted_next_features, next_state_features, reduction='none'
            ).mean(dim=1)
            
            # 缩放内在奖励
            intrinsic_rewards = prediction_error * self.intrinsic_reward_coef
            
            return intrinsic_rewards
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, 
               next_states: torch.Tensor) -> Tuple[float, float, float]:
        """更新ICM网络"""
        self.update_count += 1
        
        # 只在指定频率下更新
        if self.update_count % self.update_frequency != 0:
            return 0.0, 0.0, 0.0
        
        # 提取特征
        state_features = self.feature_network(states)
        next_state_features = self.feature_network(next_states)
        
        # 逆向模型损失
        predicted_actions = self.inverse_model(state_features, next_state_features)
        inverse_loss = self.cross_entropy_loss(predicted_actions, actions)
        
        # 前向模型损失
        predicted_next_features = self.forward_model(state_features, actions)
        forward_loss = self.mse_loss(predicted_next_features, next_state_features.detach())
        
        # 总损失
        total_loss = (self.inverse_loss_coef * inverse_loss + 
                     self.forward_loss_coef * forward_loss)
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.feature_network.parameters()) + 
            list(self.inverse_model.parameters()) + 
            list(self.forward_model.parameters()), 
            0.5
        )
        
        self.optimizer.step()
        
        return total_loss.item(), forward_loss.item(), inverse_loss.item()
    
    def get_action_prediction_accuracy(self, states: torch.Tensor, actions: torch.Tensor, 
                                     next_states: torch.Tensor) -> float:
        """计算动作预测准确率"""
        with torch.no_grad():
            state_features = self.feature_network(states)
            next_state_features = self.feature_network(next_states)
            predicted_actions = self.inverse_model(state_features, next_state_features)
            
            predicted_action_indices = torch.argmax(predicted_actions, dim=1)
            accuracy = (predicted_action_indices == actions).float().mean().item()
            
            return accuracy
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'feature_network_state_dict': self.feature_network.state_dict(),
            'inverse_model_state_dict': self.inverse_model.state_dict(),
            'forward_model_state_dict': self.forward_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.feature_network.load_state_dict(checkpoint['feature_network_state_dict'])
        self.inverse_model.load_state_dict(checkpoint['inverse_model_state_dict'])
        self.forward_model.load_state_dict(checkpoint['forward_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)


class NGU_ICM(ICM):
    """Never Give Up风格的ICM，结合episodic记忆"""
    
    def __init__(self, config, device: torch.device, num_actions: int = 18):
        super().__init__(config, device, num_actions)
        
        # Episodic记忆参数
        self.episodic_memory_size = config.get('icm.episodic_memory_size', 30000)
        self.cluster_distance = config.get('icm.cluster_distance', 0.008)
        self.c_constant = config.get('icm.c_constant', 0.001)
        self.sm_constant = config.get('icm.sm_constant', 8.0)
        
        # Episodic记忆存储
        self.episodic_memory = []
        self.episodic_counts = []
    
    def add_to_episodic_memory(self, state_features: torch.Tensor):
        """添加状态特征到episodic记忆"""
        state_features = state_features.detach().cpu()
        
        for i, features in enumerate(state_features):
            # 检查是否已存在相似状态
            if len(self.episodic_memory) == 0:
                self.episodic_memory.append(features)
                self.episodic_counts.append(1)
            else:
                # 计算与现有记忆的距离
                distances = [torch.norm(features - mem_features, p=2).item() 
                           for mem_features in self.episodic_memory]
                min_distance = min(distances)
                min_index = distances.index(min_distance)
                
                if min_distance < self.cluster_distance:
                    # 更新计数
                    self.episodic_counts[min_index] += 1
                else:
                    # 添加新记忆
                    self.episodic_memory.append(features)
                    self.episodic_counts.append(1)
                    
                    # 限制记忆大小
                    if len(self.episodic_memory) > self.episodic_memory_size:
                        self.episodic_memory.pop(0)
                        self.episodic_counts.pop(0)
    
    def compute_episodic_bonus(self, state_features: torch.Tensor) -> torch.Tensor:
        """计算episodic奖励bonus"""
        if len(self.episodic_memory) == 0:
            return torch.ones(state_features.size(0), device=self.device)
        
        bonuses = []
        state_features_cpu = state_features.detach().cpu()
        
        for features in state_features_cpu:
            # 找到最近的k个记忆
            distances = [torch.norm(features - mem_features, p=2).item() 
                        for mem_features in self.episodic_memory]
            
            # 取最小距离对应的计数
            min_distance = min(distances)
            min_index = distances.index(min_distance)
            count = self.episodic_counts[min_index]
            
            # 计算bonus
            bonus = 1.0 / (np.sqrt(count) + self.c_constant)
            bonuses.append(bonus)
        
        return torch.FloatTensor(bonuses).to(self.device)
    
    def compute_intrinsic_reward(self, states: torch.Tensor, actions: torch.Tensor, 
                                next_states: torch.Tensor) -> torch.Tensor:
        """计算结合episodic memory的内在奖励"""
        with torch.no_grad():
            # 计算基础ICM奖励
            icm_reward = super().compute_intrinsic_reward(states, actions, next_states)
            
            # 计算episodic bonus
            next_state_features = self.feature_network(next_states)
            episodic_bonus = self.compute_episodic_bonus(next_state_features)
            
            # 添加到episodic记忆
            self.add_to_episodic_memory(next_state_features)
            
            # 结合两种奖励
            combined_reward = icm_reward * episodic_bonus
            
            return combined_reward


def create_icm_module(config, device: torch.device, num_actions: int = 18, 
                     use_ngu: bool = False):
    """创建ICM模块的工厂函数"""
    if use_ngu:
        return NGU_ICM(config, device, num_actions)
    else:
        return ICM(config, device, num_actions)
