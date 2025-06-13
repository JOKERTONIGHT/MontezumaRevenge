"""
基础智能体类
定义智能体的通用接口和基础功能
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
import os
import logging

from utils import Config
from models import create_dqn_network, init_weights


class BaseAgent(ABC):
    """基础智能体抽象类"""
    
    def __init__(self, config: Config, device: torch.device, num_actions: int = 18):
        self.config = config
        self.device = device
        self.num_actions = num_actions
        
        # 基础参数
        self.gamma = config.get('dqn.gamma', 0.99)
        self.batch_size = config.get('dqn.batch_size', 32)
        self.learning_rate = config.get('dqn.learning_rate', 0.0001)
        self.target_update_frequency = config.get('dqn.target_update_frequency', 10000)
        self.learning_starts = config.get('dqn.learning_starts', 50000)
        self.train_frequency = config.get('dqn.train_frequency', 4)
        
        # Epsilon-greedy参数
        self.eps_start = config.get('dqn.eps_start', 1.0)
        self.eps_end = config.get('dqn.eps_end', 0.01)
        self.eps_decay = config.get('dqn.eps_decay', 1000000)
        self.epsilon = self.eps_start
        
        # 计数器
        self.steps_done = 0
        self.update_count = 0
        
        # 网络
        network_type = config.get('network.type', 'dueling')
        self.q_network = create_dqn_network(config, num_actions, network_type).to(device)
        self.target_network = create_dqn_network(config, num_actions, network_type).to(device)
        
        # 初始化权重
        self.q_network.apply(init_weights)
        self.target_network.apply(init_weights)
        
        # 同步目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def update_epsilon(self):
        """更新epsilon值"""
        if self.steps_done < self.eps_decay:
            self.epsilon = self.eps_start - (self.eps_start - self.eps_end) * (self.steps_done / self.eps_decay)
        else:
            self.epsilon = self.eps_end
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """选择动作"""
        if eval_mode:
            return self.select_greedy_action(state)
        
        # Epsilon-greedy策略
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            return self.select_greedy_action(state)
    
    def select_greedy_action(self, state: np.ndarray) -> int:
        """贪心动作选择"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
            return action
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    @abstractmethod
    def learn(self, replay_buffer) -> Dict[str, float]:
        """学习方法，由子类实现"""
        pass
    
    @abstractmethod
    def compute_intrinsic_reward(self, state: np.ndarray, action: int, 
                               next_state: np.ndarray) -> float:
        """计算内在奖励，由子类实现"""
        pass
    
    def step(self, replay_buffer, step: int) -> Dict[str, float]:
        """执行一步训练"""
        self.steps_done = step
        self.update_epsilon()
        
        metrics = {}
        
        # 检查是否开始学习
        if step >= self.learning_starts and step % self.train_frequency == 0:
            if replay_buffer.can_sample(self.batch_size):
                learn_metrics = self.learn(replay_buffer)
                metrics.update(learn_metrics)
                self.update_count += 1
        
        # 更新目标网络
        if step % self.target_update_frequency == 0:
            self.update_target_network()
            self.logger.info(f"Target network updated at step {step}")
        
        return metrics
    
    def save(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'update_count': self.update_count,
            'epsilon': self.epsilon,
            'config': self.config.to_dict()
        }
        
        # 子类可以添加额外的状态
        additional_state = self.get_additional_state()
        checkpoint.update(additional_state)
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.update_count = checkpoint.get('update_count', 0)
        self.epsilon = checkpoint.get('epsilon', self.eps_end)
        
        # 子类可以加载额外的状态
        self.load_additional_state(checkpoint)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_additional_state(self) -> Dict[str, Any]:
        """获取额外状态信息，子类可以重写"""
        return {}
    
    def load_additional_state(self, checkpoint: Dict[str, Any]):
        """加载额外状态信息，子类可以重写"""
        pass
    
    def get_q_values(self, states: torch.Tensor) -> torch.Tensor:
        """获取Q值"""
        return self.q_network(states)
    
    def get_target_q_values(self, states: torch.Tensor) -> torch.Tensor:
        """获取目标Q值"""
        with torch.no_grad():
            return self.target_network(states)
    
    def compute_td_error(self, states: torch.Tensor, actions: torch.Tensor, 
                        rewards: torch.Tensor, next_states: torch.Tensor, 
                        dones: torch.Tensor) -> torch.Tensor:
        """计算TD误差"""
        current_q_values = self.get_q_values(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN
        next_actions = self.get_q_values(next_states).argmax(1, keepdim=True)
        next_q_values = self.get_target_q_values(next_states).gather(1, next_actions)
        
        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        td_error = current_q_values - target_q_values.detach()
        return td_error.squeeze()
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                    rewards: torch.Tensor, next_states: torch.Tensor, 
                    dones: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算DQN损失"""
        td_error = self.compute_td_error(states, actions, rewards, next_states, dones)
        
        if weights is not None:
            loss = (td_error.pow(2) * weights).mean()
        else:
            loss = td_error.pow(2).mean()
        
        return loss, td_error.abs()
    
    def evaluate(self, env, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """评估智能体性能"""
        self.q_network.eval()
        
        episode_rewards = []
        episode_steps = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_step = 0
            done = False
            
            while not done:
                action = self.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_step += 1
                state = next_state
                
                if render:
                    env.render()
            
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
        
        self.q_network.train()
        
        return {
            'eval_mean_reward': np.mean(episode_rewards),
            'eval_std_reward': np.std(episode_rewards),
            'eval_max_reward': np.max(episode_rewards),
            'eval_min_reward': np.min(episode_rewards),
            'eval_mean_steps': np.mean(episode_steps),
            'eval_std_steps': np.std(episode_steps)
        }
    
    def get_network_stats(self) -> Dict[str, float]:
        """获取网络统计信息"""
        q_network_grad_norm = 0
        q_network_weight_norm = 0
        
        for param in self.q_network.parameters():
            if param.grad is not None:
                q_network_grad_norm += param.grad.data.norm(2).item() ** 2
            q_network_weight_norm += param.data.norm(2).item() ** 2
        
        return {
            'q_network_grad_norm': q_network_grad_norm ** 0.5,
            'q_network_weight_norm': q_network_weight_norm ** 0.5
        }


class PrioritizedAgent(BaseAgent):
    """支持优先级经验回放的智能体基类"""
    
    def __init__(self, config: Config, device: torch.device, num_actions: int = 18):
        super().__init__(config, device, num_actions)
        
        # 优先级经验回放参数
        self.use_prioritized_replay = config.get('dqn.use_prioritized_replay', False)
        self.priority_alpha = config.get('dqn.priority_alpha', 0.6)
        self.priority_beta_start = config.get('dqn.priority_beta_start', 0.4)
        self.priority_beta_frames = config.get('dqn.priority_beta_frames', 100000)
    
    def get_priority_beta(self) -> float:
        """获取当前的beta值"""
        if self.steps_done >= self.priority_beta_frames:
            return 1.0
        else:
            return self.priority_beta_start + (1.0 - self.priority_beta_start) * \
                   (self.steps_done / self.priority_beta_frames)
    
    def learn_with_priorities(self, replay_buffer) -> Dict[str, float]:
        """使用优先级经验回放学习"""
        if not self.use_prioritized_replay:
            return self.learn(replay_buffer)
        
        # 采样批次数据
        states, actions, rewards, next_states, dones, weights, indices = \
            replay_buffer.sample(self.batch_size)
        
        # 计算损失和TD误差
        loss, td_errors = self.compute_loss(states, actions, rewards, next_states, dones, weights)
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # 更新优先级
        replay_buffer.update_priorities(indices, td_errors)
        
        # 额外的学习步骤（由子类实现）
        additional_metrics = self.additional_learning_step(states, actions, next_states)
        
        metrics = {
            'q_loss': loss.item(),
            'td_error_mean': td_errors.mean().item(),
            'priority_beta': self.get_priority_beta()
        }
        metrics.update(additional_metrics)
        
        return metrics
    
    @abstractmethod
    def additional_learning_step(self, states: torch.Tensor, actions: torch.Tensor, 
                               next_states: torch.Tensor) -> Dict[str, float]:
        """额外的学习步骤，由子类实现"""
        pass
