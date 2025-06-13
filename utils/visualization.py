"""
可视化工具模块
提供训练过程的可视化和结果分析功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 数据存储
        self.metrics = defaultdict(list)
        self.episode_data = defaultdict(list)
    
    def log_episode(self, episode: int, reward: float, steps: int, 
                   epsilon: float = None, intrinsic_reward: float = None,
                   **kwargs):
        """记录单局数据"""
        self.episode_data['episode'].append(episode)
        self.episode_data['reward'].append(reward)
        self.episode_data['steps'].append(steps)
        
        if epsilon is not None:
            self.episode_data['epsilon'].append(epsilon)
        if intrinsic_reward is not None:
            self.episode_data['intrinsic_reward'].append(intrinsic_reward)
        
        # 记录其他指标
        for key, value in kwargs.items():
            self.episode_data[key].append(value)
    
    def log_training_metrics(self, step: int, **metrics):
        """记录训练指标"""
        self.metrics['step'].append(step)
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def plot_training_progress(self, window_size: int = 100, save: bool = True):
        """绘制训练进度图"""
        if not self.episode_data['episode']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        episodes = self.episode_data['episode']
        rewards = self.episode_data['reward']
        
        # 奖励曲线
        axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue', linewidth=0.5)
        if len(rewards) >= window_size:
            smoothed_rewards = self._smooth_curve(rewards, window_size)
            axes[0, 0].plot(episodes, smoothed_rewards, color='red', linewidth=2, 
                           label=f'Moving Average ({window_size})')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 步数曲线
        if 'steps' in self.episode_data:
            steps = self.episode_data['steps']
            axes[0, 1].plot(episodes, steps, alpha=0.6, color='green')
            if len(steps) >= window_size:
                smoothed_steps = self._smooth_curve(steps, window_size)
                axes[0, 1].plot(episodes, smoothed_steps, color='darkgreen', 
                               linewidth=2, label=f'Moving Average ({window_size})')
            axes[0, 1].set_title('Episode Steps')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Epsilon衰减
        if 'epsilon' in self.episode_data:
            epsilon = self.episode_data['epsilon']
            axes[1, 0].plot(episodes, epsilon, color='orange', linewidth=2)
            axes[1, 0].set_title('Epsilon Decay')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].grid(True)
        
        # 内在奖励
        if 'intrinsic_reward' in self.episode_data:
            intrinsic_rewards = self.episode_data['intrinsic_reward']
            axes[1, 1].plot(episodes, intrinsic_rewards, alpha=0.6, color='purple')
            if len(intrinsic_rewards) >= window_size:
                smoothed_intrinsic = self._smooth_curve(intrinsic_rewards, window_size)
                axes[1, 1].plot(episodes, smoothed_intrinsic, color='darkviolet', 
                               linewidth=2, label=f'Moving Average ({window_size})')
            axes[1, 1].set_title('Intrinsic Rewards')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Intrinsic Reward')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_loss_curves(self, save: bool = True):
        """绘制损失曲线"""
        if not self.metrics['step']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Losses', fontsize=16)
        
        steps = self.metrics['step']
        
        # Q损失
        if 'q_loss' in self.metrics:
            axes[0, 0].plot(steps, self.metrics['q_loss'], color='red', linewidth=1)
            axes[0, 0].set_title('Q-Network Loss')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # RND损失
        if 'rnd_loss' in self.metrics:
            axes[0, 1].plot(steps, self.metrics['rnd_loss'], color='blue', linewidth=1)
            axes[0, 1].set_title('RND Loss')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # ICM损失
        if 'icm_loss' in self.metrics:
            axes[1, 0].plot(steps, self.metrics['icm_loss'], color='green', linewidth=1)
            axes[1, 0].set_title('ICM Total Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # 前向损失
        if 'forward_loss' in self.metrics:
            axes[1, 1].plot(steps, self.metrics['forward_loss'], color='orange', 
                           linewidth=1, label='Forward Loss')
        if 'inverse_loss' in self.metrics:
            axes[1, 1].plot(steps, self.metrics['inverse_loss'], color='purple', 
                           linewidth=1, label='Inverse Loss')
        if 'forward_loss' in self.metrics or 'inverse_loss' in self.metrics:
            axes[1, 1].set_title('ICM Component Losses')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'loss_curves.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_reward_distribution(self, save: bool = True):
        """绘制奖励分布图"""
        if not self.episode_data['reward']:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Reward Analysis', fontsize=16)
        
        rewards = self.episode_data['reward']
        
        # 奖励直方图
        axes[0].hist(rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(rewards):.2f}')
        axes[0].axvline(np.median(rewards), color='green', linestyle='--', 
                       label=f'Median: {np.median(rewards):.2f}')
        axes[0].set_title('Reward Distribution')
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True)
        
        # 奖励箱线图（按训练阶段分组）
        if len(rewards) > 100:
            num_groups = 5
            group_size = len(rewards) // num_groups
            grouped_rewards = []
            labels = []
            
            for i in range(num_groups):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < num_groups - 1 else len(rewards)
                grouped_rewards.append(rewards[start_idx:end_idx])
                labels.append(f'Episodes\n{start_idx}-{end_idx}')
            
            axes[1].boxplot(grouped_rewards, labels=labels)
            axes[1].set_title('Reward by Training Stage')
            axes[1].set_xlabel('Training Stage')
            axes[1].set_ylabel('Reward')
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'reward_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison(self, other_visualizer: 'TrainingVisualizer', 
                       labels: List[str], save: bool = True):
        """比较两个训练过程"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Comparison', fontsize=16)
        
        # 奖励比较
        if self.episode_data['reward'] and other_visualizer.episode_data['reward']:
            episodes1 = self.episode_data['episode']
            rewards1 = self._smooth_curve(self.episode_data['reward'], 100)
            episodes2 = other_visualizer.episode_data['episode']
            rewards2 = other_visualizer._smooth_curve(other_visualizer.episode_data['reward'], 100)
            
            axes[0, 0].plot(episodes1, rewards1, label=labels[0], linewidth=2)
            axes[0, 0].plot(episodes2, rewards2, label=labels[1], linewidth=2)
            axes[0, 0].set_title('Reward Comparison')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Smoothed Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 添加更多比较图...
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'training_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _smooth_curve(self, data: List[float], window_size: int) -> List[float]:
        """平滑曲线"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            smoothed.append(np.mean(data[start_idx:end_idx]))
        
        return smoothed
    
    def save_data(self, filename: str = "training_data.npz"):
        """保存训练数据"""
        save_path = os.path.join(self.save_dir, filename)
        
        # 转换为numpy数组
        episode_arrays = {}
        for key, values in self.episode_data.items():
            episode_arrays[f"episode_{key}"] = np.array(values)
        
        metric_arrays = {}
        for key, values in self.metrics.items():
            metric_arrays[f"metric_{key}"] = np.array(values)
        
        # 合并并保存
        all_arrays = {**episode_arrays, **metric_arrays}
        np.savez(save_path, **all_arrays)
        
        print(f"Training data saved to {save_path}")
    
    def load_data(self, filename: str = "training_data.npz"):
        """加载训练数据"""
        load_path = os.path.join(self.save_dir, filename)
        
        if not os.path.exists(load_path):
            print(f"File {load_path} not found")
            return
        
        data = np.load(load_path)
        
        # 恢复数据
        for key in data.keys():
            if key.startswith("episode_"):
                actual_key = key[8:]  # 移除 "episode_" 前缀
                self.episode_data[actual_key] = data[key].tolist()
            elif key.startswith("metric_"):
                actual_key = key[7:]  # 移除 "metric_" 前缀
                self.metrics[actual_key] = data[key].tolist()
        
        print(f"Training data loaded from {load_path}")


def plot_network_weights(model: torch.nn.Module, save_path: str = None):
    """可视化网络权重分布"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Network Weight Distributions', fontsize=16)
    
    all_weights = []
    layer_weights = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weights = param.data.cpu().numpy().flatten()
            all_weights.extend(weights)
            layer_weights[name] = weights
    
    # 所有权重分布
    axes[0, 0].hist(all_weights, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('All Weights Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True)
    
    # 权重统计
    axes[0, 1].boxplot([weights for weights in layer_weights.values()], 
                      labels=[name.split('.')[0] for name in layer_weights.keys()])
    axes[0, 1].set_title('Weights by Layer')
    axes[0, 1].set_ylabel('Weight Value')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_training_report(visualizer: TrainingVisualizer, 
                          config: Dict, 
                          final_metrics: Dict = None):
    """生成训练报告"""
    report_path = os.path.join(visualizer.save_dir, "training_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 训练报告\n\n")
        
        # 配置信息
        f.write("## 实验配置\n\n")
        f.write(f"- 环境: {config.get('environment', {}).get('name', 'Unknown')}\n")
        f.write(f"- 算法: {config.get('experiment', {}).get('agent_type', 'Unknown')}\n")
        f.write(f"- 总训练步数: {config.get('training', {}).get('total_timesteps', 'Unknown')}\n")
        f.write(f"- 学习率: {config.get('dqn', {}).get('learning_rate', 'Unknown')}\n")
        f.write(f"- 批量大小: {config.get('dqn', {}).get('batch_size', 'Unknown')}\n\n")
        
        # 训练统计
        if visualizer.episode_data['reward']:
            rewards = visualizer.episode_data['reward']
            f.write("## 训练统计\n\n")
            f.write(f"- 总局数: {len(rewards)}\n")
            f.write(f"- 平均奖励: {np.mean(rewards):.2f}\n")
            f.write(f"- 最大奖励: {np.max(rewards):.2f}\n")
            f.write(f"- 最小奖励: {np.min(rewards):.2f}\n")
            f.write(f"- 奖励标准差: {np.std(rewards):.2f}\n\n")
        
        # 最终指标
        if final_metrics:
            f.write("## 最终性能\n\n")
            for key, value in final_metrics.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
        
        f.write("## 可视化图表\n\n")
        f.write("- [训练进度](training_progress.png)\n")
        f.write("- [损失曲线](loss_curves.png)\n")
        f.write("- [奖励分析](reward_analysis.png)\n")
    
    print(f"训练报告已保存到: {report_path}")
