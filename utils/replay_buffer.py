"""
经验回放缓冲区模块
实现优先级经验回放和标准经验回放
"""

import numpy as np
import torch
import random
from collections import namedtuple, deque
from typing import List, Tuple


# 经验元组定义
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """标准经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # 预分配内存
        self.states = np.zeros((capacity,) + state_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity,) + state_shape, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """采样批次经验"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.BoolTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size
    
    def can_sample(self, batch_size: int) -> bool:
        """检查是否可以采样"""
        return self.size >= batch_size


class SumTree:
    """求和树数据结构，用于优先级经验回放"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """向上传播变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """检索叶子节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """获取总优先级"""
        return self.tree[0]
    
    def add(self, priority: float, data):
        """添加数据"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """获取数据"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_shape: tuple, device: torch.device, 
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
        # 预分配内存
        self.state_shape = state_shape
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        experience = Experience(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """采样批次经验"""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        weights /= weights.max()
        
        # 转换为张量
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: torch.Tensor):
        """更新优先级"""
        td_errors = td_errors.detach().cpu().numpy()
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.tree.n_entries
    
    def can_sample(self, batch_size: int) -> bool:
        """检查是否可以采样"""
        return self.tree.n_entries >= batch_size


class EpisodeBuffer:
    """单局游戏缓冲区，用于存储内在奖励计算需要的完整轨迹"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.intrinsic_rewards = []
        self.dones = []
    
    def push(self, state, action, reward, done, intrinsic_reward=0.0):
        """添加步骤"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.intrinsic_rewards.append(intrinsic_reward)
        self.dones.append(done)
    
    def get_episode(self) -> Tuple[List, ...]:
        """获取完整轨迹"""
        return (self.states.copy(), self.actions.copy(), 
                self.rewards.copy(), self.intrinsic_rewards.copy(), 
                self.dones.copy())
    
    def __len__(self):
        return len(self.states)


class RewardNormalizer:
    """奖励标准化器"""
    
    def __init__(self, epsilon=1e-8, cliprange=5.0):
        self.epsilon = epsilon
        self.cliprange = cliprange
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0
    
    def update(self, x):
        """更新统计信息"""
        self.count += 1
        delta = x - self.running_mean
        self.running_mean += delta / self.count
        delta2 = x - self.running_mean
        self.running_var += delta * delta2
    
    def normalize(self, x):
        """标准化奖励"""
        if self.count < 2:
            return x
        
        var = self.running_var / (self.count - 1)
        std = np.sqrt(var + self.epsilon)
        normalized = x / std
        
        return np.clip(normalized, -self.cliprange, self.cliprange)
    
    def __call__(self, x, update_stats=True):
        if update_stats:
            self.update(x)
        return self.normalize(x)
