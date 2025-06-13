"""
DQN + ICM 智能体
结合Deep Q-Network和Intrinsic Curiosity Module的智能体实现
"""

import torch
import numpy as np
from typing import Dict, Tuple
import logging

from .base_agent import BaseAgent, PrioritizedAgent
from models import create_icm_module
from utils import Config, RewardNormalizer


class DQNICMAgent(PrioritizedAgent):
    """DQN + ICM 智能体"""
    
    def __init__(self, config: Config, device: torch.device, num_actions: int = 18):
        super().__init__(config, device, num_actions)
        
        # 创建ICM模块
        self.icm_module = create_icm_module(config, device, num_actions, use_ngu=False)
        
        # 内在奖励相关参数
        self.use_intrinsic_reward = config.get('icm.use_intrinsic_reward', True)
        self.intrinsic_reward_coef = config.get('icm.intrinsic_reward_coef', 0.01)
        self.normalize_intrinsic_reward = config.get('icm.normalize_intrinsic_reward', True)
        
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
        
        # ICM性能统计
        self.icm_stats = {
            'action_prediction_accuracy': 0.0,
            'forward_loss': 0.0,
            'inverse_loss': 0.0
        }
        
        self.logger = logging.getLogger("DQNICMAgent")
    
    def compute_intrinsic_reward(self, state: np.ndarray, action: int, 
                               next_state: np.ndarray) -> float:
        """计算内在奖励"""
        if not self.use_intrinsic_reward:
            return 0.0
        
        # 转换为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # 计算ICM内在奖励
        intrinsic_reward = self.icm_module.compute_intrinsic_reward(
            state_tensor, action_tensor, next_state_tensor
        )
        intrinsic_reward_value = intrinsic_reward.item()
        
        # 标准化内在奖励
        if self.intrinsic_reward_normalizer is not None:
            intrinsic_reward_value = self.intrinsic_reward_normalizer(
                intrinsic_reward_value, update_stats=True
            )
        
        # 更新统计信息
        self.update_intrinsic_reward_stats(intrinsic_reward_value)
        
        return intrinsic_reward_value
    
    def compute_batch_intrinsic_rewards(self, states: torch.Tensor, actions: torch.Tensor,
                                      next_states: torch.Tensor) -> torch.Tensor:
        """批量计算内在奖励"""
        if not self.use_intrinsic_reward:
            return torch.zeros(states.size(0), device=self.device)
        
        intrinsic_rewards = self.icm_module.compute_intrinsic_reward(states, actions, next_states)
        
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
        intrinsic_rewards = self.compute_batch_intrinsic_rewards(states, actions, next_states)
        combined_rewards = rewards + intrinsic_rewards
        
        # 计算DQN损失
        loss, td_errors = self.compute_loss(states, actions, combined_rewards, next_states, dones)
        
        # 更新Q网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # 更新ICM模块
        icm_total_loss, forward_loss, inverse_loss = self.icm_module.update(states, actions, next_states)
        
        # 计算动作预测准确率
        action_accuracy = self.icm_module.get_action_prediction_accuracy(states, actions, next_states)
        
        # 更新ICM统计
        self.icm_stats['action_prediction_accuracy'] = action_accuracy
        self.icm_stats['forward_loss'] = forward_loss
        self.icm_stats['inverse_loss'] = inverse_loss
        
        metrics = {
            'q_loss': loss.item(),
            'icm_loss': icm_total_loss,
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss,
            'action_prediction_accuracy': action_accuracy,
            'td_error_mean': td_errors.mean().item(),
            'intrinsic_reward_mean': intrinsic_rewards.mean().item(),
            'intrinsic_reward_std': intrinsic_rewards.std().item(),
            'combined_reward_mean': combined_rewards.mean().item()
        }
        
        return metrics
    
    def additional_learning_step(self, states: torch.Tensor, actions: torch.Tensor, 
                               next_states: torch.Tensor) -> Dict[str, float]:
        """优先级经验回放的额外学习步骤"""
        # 更新ICM模块
        icm_total_loss, forward_loss, inverse_loss = self.icm_module.update(states, actions, next_states)
        
        # 计算内在奖励统计
        intrinsic_rewards = self.compute_batch_intrinsic_rewards(states, actions, next_states)
        
        # 计算动作预测准确率
        action_accuracy = self.icm_module.get_action_prediction_accuracy(states, actions, next_states)
        
        # 更新ICM统计
        self.icm_stats['action_prediction_accuracy'] = action_accuracy
        self.icm_stats['forward_loss'] = forward_loss
        self.icm_stats['inverse_loss'] = inverse_loss
        
        return {
            'icm_loss': icm_total_loss,
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss,
            'action_prediction_accuracy': action_accuracy,
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
        
        # 添加ICM相关统计
        icm_stats = {
            'epsilon': self.epsilon,
            'intrinsic_reward_coef': self.intrinsic_reward_coef,
            'intrinsic_reward_stats_mean': self.intrinsic_reward_stats['mean'],
            'intrinsic_reward_stats_std': self.intrinsic_reward_stats['std'],
            'icm_action_accuracy': self.icm_stats['action_prediction_accuracy'],
            'icm_forward_loss': self.icm_stats['forward_loss'],
            'icm_inverse_loss': self.icm_stats['inverse_loss']
        }
        metrics.update(icm_stats)
        
        return metrics
    
    def get_additional_state(self) -> Dict:
        """获取额外状态信息"""
        return {
            'intrinsic_reward_stats': self.intrinsic_reward_stats,
            'intrinsic_reward_normalizer': self.intrinsic_reward_normalizer,
            'icm_stats': self.icm_stats,
            'icm_module_state': None  # ICM模块有自己的保存方法
        }
    
    def load_additional_state(self, checkpoint: Dict):
        """加载额外状态信息"""
        if 'intrinsic_reward_stats' in checkpoint:
            self.intrinsic_reward_stats = checkpoint['intrinsic_reward_stats']
        
        if 'intrinsic_reward_normalizer' in checkpoint:
            self.intrinsic_reward_normalizer = checkpoint['intrinsic_reward_normalizer']
        
        if 'icm_stats' in checkpoint:
            self.icm_stats = checkpoint['icm_stats']
    
    def save(self, filepath: str):
        """保存模型和ICM模块"""
        # 保存主要模型
        super().save(filepath)
        
        # 保存ICM模块
        icm_filepath = filepath.replace('.pth', '_icm.pth')
        self.icm_module.save(icm_filepath)
        
        self.logger.info(f"DQN+ICM model saved to {filepath} and {icm_filepath}")
    
    def load(self, filepath: str):
        """加载模型和ICM模块"""
        # 加载主要模型
        super().load(filepath)
        
        # 加载ICM模块
        icm_filepath = filepath.replace('.pth', '_icm.pth')
        try:
            self.icm_module.load(icm_filepath)
            self.logger.info(f"ICM module loaded from {icm_filepath}")
        except FileNotFoundError:
            self.logger.warning(f"ICM module file {icm_filepath} not found, using initialized weights")
    
    def evaluate_with_intrinsic_rewards(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """评估时包含内在奖励信息"""
        self.q_network.eval()
        
        episode_rewards = []
        episode_intrinsic_rewards = []
        episode_steps = []
        total_action_accuracy = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_intrinsic_reward = 0
            episode_step = 0
            episode_accuracy = 0
            done = False
            
            while not done:
                action = self.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 计算内在奖励
                intrinsic_reward = self.compute_intrinsic_reward(state, action, next_state)
                
                # 计算动作预测准确率（用于分析）
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor([action]).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                accuracy = self.icm_module.get_action_prediction_accuracy(
                    state_tensor, action_tensor, next_state_tensor
                )
                
                episode_reward += reward
                episode_intrinsic_reward += intrinsic_reward
                episode_accuracy += accuracy
                episode_step += 1
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_intrinsic_rewards.append(episode_intrinsic_reward)
            episode_steps.append(episode_step)
            total_action_accuracy += episode_accuracy / episode_step if episode_step > 0 else 0
        
        self.q_network.train()
        
        eval_metrics = {
            'eval_mean_reward': np.mean(episode_rewards),
            'eval_std_reward': np.std(episode_rewards),
            'eval_mean_intrinsic_reward': np.mean(episode_intrinsic_rewards),
            'eval_std_intrinsic_reward': np.std(episode_intrinsic_rewards),
            'eval_mean_steps': np.mean(episode_steps),
            'eval_action_prediction_accuracy': total_action_accuracy / num_episodes
        }
        
        return eval_metrics
    
    def get_curiosity_level(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """获取状态-动作的好奇心水平（用于分析）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        intrinsic_reward = self.icm_module.compute_intrinsic_reward(
            state_tensor, action_tensor, next_state_tensor
        )
        return intrinsic_reward.item()
    
    def get_feature_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """计算两个状态特征的相似度"""
        state1_tensor = torch.FloatTensor(state1).unsqueeze(0).to(self.device)
        state2_tensor = torch.FloatTensor(state2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features1 = self.icm_module.feature_network(state1_tensor)
            features2 = self.icm_module.feature_network(state2_tensor)
            
            # 计算余弦相似度
            similarity = torch.cosine_similarity(features1, features2, dim=1)
            return similarity.item()
    
    def reset_episode_stats(self):
        """重置局内统计信息"""
        # 重置ICM的episodic memory（如果使用NGU版本）
        if hasattr(self.icm_module, 'episodic_memory'):
            # 对于NGU版本，可以选择性地清理episodic memory
            pass
    
    def analyze_exploration_efficiency(self, states_visited: list) -> Dict[str, float]:
        """分析探索效率"""
        if len(states_visited) < 2:
            return {'exploration_efficiency': 0.0, 'state_diversity': 0.0}
        
        # 计算状态多样性
        similarities = []
        for i in range(len(states_visited)):
            for j in range(i + 1, min(i + 10, len(states_visited))):  # 只比较邻近状态
                similarity = self.get_feature_similarity(states_visited[i], states_visited[j])
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 1.0
        state_diversity = 1.0 - avg_similarity  # 相似度越低，多样性越高
        
        # 探索效率（状态多样性的变种）
        exploration_efficiency = state_diversity
        
        return {
            'exploration_efficiency': exploration_efficiency,
            'state_diversity': state_diversity,
            'avg_state_similarity': avg_similarity
        }
