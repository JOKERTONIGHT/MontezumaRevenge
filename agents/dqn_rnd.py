"""
DQN + RND 智能体
结合Deep Q-Network和Random Network Distillation的智能体实现
"""

import torch
import numpy as np
from typing import Dict, Tuple
import logging

from .base_agent import BaseAgent, PrioritizedAgent
from models import create_rnd_module
from utils import Config, RewardNormalizer


class DQNRNDAgent(PrioritizedAgent):
    """DQN + RND 智能体"""
    
    def __init__(self, config: Config, device: torch.device, num_actions: int = 18):
        super().__init__(config, device, num_actions)
        
        # 创建RND模块
        self.rnd_module = create_rnd_module(config, device, adaptive=True)
        
        # 内在奖励相关参数
        self.use_intrinsic_reward = config.get('rnd.use_intrinsic_reward', True)
        self.intrinsic_reward_coef = config.get('rnd.intrinsic_reward_coef', 1.0)
        self.normalize_intrinsic_reward = config.get('rnd.normalize_intrinsic_reward', True)
        
        # 奖励标准化器
        if self.normalize_intrinsic_reward:
            self.intrinsic_reward_normalizer = RewardNormalizer()
        else:
            self.intrinsic_reward_normalizer = None
        
        # 统计信息
        self.intrinsic_reward_stats = {
            'mean': 0.0,
            'std': 0.0,
            'count': 0
        }
        
        self.logger = logging.getLogger("DQNRNDAgent")
    
    def compute_intrinsic_reward(self, state: np.ndarray, action: int, 
                               next_state: np.ndarray) -> float:
        """计算内在奖励"""
        if not self.use_intrinsic_reward:
            return 0.0
        
        # 转换为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # 计算RND内在奖励
        intrinsic_reward = self.rnd_module.compute_intrinsic_reward(next_state_tensor)
        intrinsic_reward_value = intrinsic_reward.item()
        
        # 标准化内在奖励
        if self.intrinsic_reward_normalizer is not None:
            intrinsic_reward_value = self.intrinsic_reward_normalizer(
                intrinsic_reward_value, update_stats=True
            )
        
        # 更新统计信息
        self.update_intrinsic_reward_stats(intrinsic_reward_value)
        
        return intrinsic_reward_value
    
    def compute_batch_intrinsic_rewards(self, next_states: torch.Tensor) -> torch.Tensor:
        """批量计算内在奖励"""
        if not self.use_intrinsic_reward:
            return torch.zeros(next_states.size(0), device=self.device)
        
        intrinsic_rewards = self.rnd_module.compute_intrinsic_reward(next_states)
        
        # 批量标准化
        if self.intrinsic_reward_normalizer is not None:
            intrinsic_rewards_np = intrinsic_rewards.cpu().numpy()
            normalized_rewards = []
            for reward in intrinsic_rewards_np:
                normalized_reward = self.intrinsic_reward_normalizer(reward, update_stats=False)
                normalized_rewards.append(normalized_reward)
            intrinsic_rewards = torch.FloatTensor(normalized_rewards).to(self.device)
        
        return intrinsic_rewards
    
    def update_intrinsic_reward_stats(self, intrinsic_reward: float):
        """更新内在奖励统计信息"""
        self.intrinsic_reward_stats['count'] += 1
        delta = intrinsic_reward - self.intrinsic_reward_stats['mean']
        self.intrinsic_reward_stats['mean'] += delta / self.intrinsic_reward_stats['count']
        delta2 = intrinsic_reward - self.intrinsic_reward_stats['mean']
        
        if self.intrinsic_reward_stats['count'] > 1:
            variance = ((self.intrinsic_reward_stats['count'] - 2) * (self.intrinsic_reward_stats['std'] ** 2) + 
                       delta * delta2) / (self.intrinsic_reward_stats['count'] - 1)
            self.intrinsic_reward_stats['std'] = np.sqrt(variance)
    
    def learn(self, replay_buffer) -> Dict[str, float]:
        """学习方法"""
        if self.use_prioritized_replay:
            return self.learn_with_priorities(replay_buffer)
        else:
            return self.learn_standard(replay_buffer)
    
    def learn_standard(self, replay_buffer) -> Dict[str, float]:
        """标准经验回放学习"""
        # 采样批次数据
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        
        # 计算内在奖励
        intrinsic_rewards = self.compute_batch_intrinsic_rewards(next_states)
        combined_rewards = rewards + intrinsic_rewards
        
        # 计算DQN损失
        loss, td_errors = self.compute_loss(states, actions, combined_rewards, next_states, dones)
        
        # 更新Q网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # 更新RND模块
        rnd_loss = self.rnd_module.update(next_states)
        
        metrics = {
            'q_loss': loss.item(),
            'rnd_loss': rnd_loss,
            'td_error_mean': td_errors.mean().item(),
            'intrinsic_reward_mean': intrinsic_rewards.mean().item(),
            'intrinsic_reward_std': intrinsic_rewards.std().item(),
            'combined_reward_mean': combined_rewards.mean().item()
        }
        
        return metrics
    
    def additional_learning_step(self, states: torch.Tensor, actions: torch.Tensor, 
                               next_states: torch.Tensor) -> Dict[str, float]:
        """优先级经验回放的额外学习步骤"""
        # 更新RND模块
        rnd_loss = self.rnd_module.update(next_states)
        
        # 计算内在奖励统计
        intrinsic_rewards = self.compute_batch_intrinsic_rewards(next_states)
        
        return {
            'rnd_loss': rnd_loss,
            'intrinsic_reward_mean': intrinsic_rewards.mean().item(),
            'intrinsic_reward_std': intrinsic_rewards.std().item()
        }
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                    rewards: torch.Tensor, next_states: torch.Tensor, 
                    dones: torch.Tensor, weights=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算DQN损失（重写以支持内在奖励）"""
        return super().compute_loss(states, actions, rewards, next_states, dones, weights)
    
    def step(self, replay_buffer, step: int) -> Dict[str, float]:
        """执行一步训练"""
        metrics = super().step(replay_buffer, step)
        
        # 添加RND相关统计
        rnd_stats = {
            'epsilon': self.epsilon,
            'intrinsic_reward_coef': self.rnd_module.intrinsic_reward_coef,
            'intrinsic_reward_stats_mean': self.intrinsic_reward_stats['mean'],
            'intrinsic_reward_stats_std': self.intrinsic_reward_stats['std']
        }
        metrics.update(rnd_stats)
        
        return metrics
    
    def get_additional_state(self) -> Dict:
        """获取额外状态信息"""
        return {
            'intrinsic_reward_stats': self.intrinsic_reward_stats,
            'intrinsic_reward_normalizer': self.intrinsic_reward_normalizer,
            'rnd_module_state': None  # RND模块有自己的保存方法
        }
    
    def load_additional_state(self, checkpoint: Dict):
        """加载额外状态信息"""
        if 'intrinsic_reward_stats' in checkpoint:
            self.intrinsic_reward_stats = checkpoint['intrinsic_reward_stats']
        
        if 'intrinsic_reward_normalizer' in checkpoint:
            self.intrinsic_reward_normalizer = checkpoint['intrinsic_reward_normalizer']
    
    def save(self, filepath: str):
        """保存模型和RND模块"""
        # 保存主要模型
        super().save(filepath)
        
        # 保存RND模块
        rnd_filepath = filepath.replace('.pth', '_rnd.pth')
        self.rnd_module.save(rnd_filepath)
        
        self.logger.info(f"DQN+RND model saved to {filepath} and {rnd_filepath}")
    
    def load(self, filepath: str):
        """加载模型和RND模块"""
        # 加载主要模型
        super().load(filepath)
        
        # 加载RND模块
        rnd_filepath = filepath.replace('.pth', '_rnd.pth')
        try:
            self.rnd_module.load(rnd_filepath)
            self.logger.info(f"RND module loaded from {rnd_filepath}")
        except FileNotFoundError:
            self.logger.warning(f"RND module file {rnd_filepath} not found, using initialized weights")
    
    def evaluate_with_intrinsic_rewards(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """评估时包含内在奖励信息"""
        self.q_network.eval()
        
        episode_rewards = []
        episode_intrinsic_rewards = []
        episode_steps = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_intrinsic_reward = 0
            episode_step = 0
            done = False
            
            while not done:
                action = self.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 计算内在奖励
                intrinsic_reward = self.compute_intrinsic_reward(state, action, next_state)
                
                episode_reward += reward
                episode_intrinsic_reward += intrinsic_reward
                episode_step += 1
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_intrinsic_rewards.append(episode_intrinsic_reward)
            episode_steps.append(episode_step)
        
        self.q_network.train()
        
        eval_metrics = {
            'eval_mean_reward': np.mean(episode_rewards),
            'eval_std_reward': np.std(episode_rewards),
            'eval_mean_intrinsic_reward': np.mean(episode_intrinsic_rewards),
            'eval_std_intrinsic_reward': np.std(episode_intrinsic_rewards),
            'eval_mean_steps': np.mean(episode_steps)
        }
        
        return eval_metrics
    
    def get_exploration_bonus(self, state: np.ndarray) -> float:
        """获取状态的探索bonus（用于分析）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        intrinsic_reward = self.rnd_module.compute_intrinsic_reward(state_tensor)
        return intrinsic_reward.item()
    
    def reset_episode_stats(self):
        """重置局内统计信息"""
        # 可以用于重置一些episodic的统计信息
        pass
    
    def adapt_exploration(self, episode_reward: float):
        """根据表现自适应调整探索"""
        if hasattr(self.rnd_module, 'adapt_coefficient'):
            self.rnd_module.adapt_coefficient(episode_reward)
