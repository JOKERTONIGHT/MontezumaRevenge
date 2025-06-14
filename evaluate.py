"""
评估脚本
评估训练好的强化学习智能体性能
"""

import os
import argparse
import logging
import time
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from typing import Dict, List

from utils import (
    Config, set_random_seed, get_device, make_atari_env,
    TrainingVisualizer
)
from agents import create_agent


def setup_logging():
    """设置评估日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("Evaluation")


def evaluate_agent(agent, env, num_episodes: int = 10, render: bool = False) -> Dict:
    """评估智能体性能"""
    logger = logging.getLogger("Evaluation")
    
    episode_rewards = []
    episode_intrinsic_rewards = []
    episode_steps = []
    episode_times = []
    state_visits = []  # 用于分析探索
    
    logger.info(f"开始评估，共 {num_episodes} 局")
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        state, _ = env.reset()
        episode_reward = 0
        episode_intrinsic_reward = 0
        episode_step = 0
        episode_states = []
        done = False
        
        while not done:
            # 选择动作（贪心策略）
            action = agent.select_action(state, eval_mode=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 计算内在奖励（用于分析）
            intrinsic_reward = agent.compute_intrinsic_reward(state, action, next_state)
            
            episode_reward += reward
            episode_intrinsic_reward += intrinsic_reward
            episode_step += 1
            episode_states.append(state.copy())
            
            state = next_state
            
            if render:
                env.render()
                time.sleep(0.02)  # 稍微减慢显示速度
        
        episode_time = time.time() - episode_start_time
        
        episode_rewards.append(episode_reward)
        episode_intrinsic_rewards.append(episode_intrinsic_reward)
        episode_steps.append(episode_step)
        episode_times.append(episode_time)
        state_visits.append(episode_states)
        
        logger.info(f"Episode {episode+1}/{num_episodes}: 奖励 = {episode_reward:.1f}, 步数 = {episode_step}")
    
    # 计算统计信息
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'mean_intrinsic_reward': np.mean(episode_intrinsic_rewards),
        'std_intrinsic_reward': np.std(episode_intrinsic_rewards),
        'mean_steps': np.mean(episode_steps),
        'std_steps': np.std(episode_steps),
        'mean_time': np.mean(episode_times),
        'total_episodes': num_episodes,
        'success_rate': sum(1 for r in episode_rewards if r > 0) / num_episodes,
        'episode_rewards': episode_rewards,
        'episode_intrinsic_rewards': episode_intrinsic_rewards,
        'episode_steps': episode_steps,
        'state_visits': state_visits
    }
    
    # 分析探索效率（如果智能体支持）
    if hasattr(agent, 'analyze_exploration_efficiency'):
        all_states = []
        for states in state_visits[:min(10, len(state_visits))]:  # 只分析前10局
            all_states.extend(states)
        if all_states:
            exploration_metrics = agent.analyze_exploration_efficiency(all_states)
            results.update(exploration_metrics)
    
    return results


def plot_evaluation_results(results: Dict, save_path: str = None):
    """绘制评估结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Agent Evaluation Results', fontsize=16)
    
    # 奖励分布
    axes[0, 0].hist(results['episode_rewards'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(results['mean_reward'], color='red', linestyle='--', 
                      label=f'Mean: {results["mean_reward"]:.2f}')
    axes[0, 0].set_title('Episode Rewards Distribution')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 奖励时间序列
    axes[0, 1].plot(results['episode_rewards'], alpha=0.7, color='blue')
    axes[0, 1].axhline(results['mean_reward'], color='red', linestyle='--', 
                      label=f'Mean: {results["mean_reward"]:.2f}')
    axes[0, 1].set_title('Episode Rewards Over Time')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 步数分布
    axes[0, 2].hist(results['episode_steps'], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 2].axvline(results['mean_steps'], color='red', linestyle='--', 
                      label=f'Mean: {results["mean_steps"]:.1f}')
    axes[0, 2].set_title('Episode Steps Distribution')
    axes[0, 2].set_xlabel('Steps')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 内在奖励分布
    axes[1, 0].hist(results['episode_intrinsic_rewards'], bins=30, alpha=0.7, 
                   color='purple', edgecolor='black')
    axes[1, 0].axvline(results['mean_intrinsic_reward'], color='red', linestyle='--', 
                      label=f'Mean: {results["mean_intrinsic_reward"]:.3f}')
    axes[1, 0].set_title('Intrinsic Rewards Distribution')
    axes[1, 0].set_xlabel('Intrinsic Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 奖励 vs 步数散点图
    axes[1, 1].scatter(results['episode_steps'], results['episode_rewards'], 
                      alpha=0.6, color='orange')
    axes[1, 1].set_title('Reward vs Steps')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 成功率和其他统计
    stats_text = f"""Statistics:
Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}
Min/Max Reward: {results['min_reward']:.1f} / {results['max_reward']:.1f}
Success Rate: {results['success_rate']:.1%}
Mean Steps: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}
Mean Intrinsic Reward: {results['mean_intrinsic_reward']:.3f}"""
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_title('Summary Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"评估结果图表保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_agents(agent_results: Dict[str, Dict], save_path: str = None):
    """比较多个智能体的性能"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Agent Comparison', fontsize=16)
    
    agent_names = list(agent_results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 奖励分布比较
    for i, (name, results) in enumerate(agent_results.items()):
        axes[0, 0].hist(results['episode_rewards'], bins=30, alpha=0.5, 
                       label=name, color=colors[i % len(colors)])
    axes[0, 0].set_title('Reward Distributions Comparison')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 平均奖励比较
    mean_rewards = [results['mean_reward'] for results in agent_results.values()]
    std_rewards = [results['std_reward'] for results in agent_results.values()]
    
    bars = axes[0, 1].bar(agent_names, mean_rewards, yerr=std_rewards, 
                         capsize=5, color=colors[:len(agent_names)])
    axes[0, 1].set_title('Mean Reward Comparison')
    axes[0, 1].set_ylabel('Mean Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 为每个柱子添加数值标签
    for bar, mean_reward in zip(bars, mean_rewards):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean_reward:.1f}', ha='center', va='bottom')
    
    # 成功率比较
    success_rates = [results['success_rate'] for results in agent_results.values()]
    bars = axes[1, 0].bar(agent_names, success_rates, color=colors[:len(agent_names)])
    axes[1, 0].set_title('Success Rate Comparison')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 为每个柱子添加百分比标签
    for bar, success_rate in zip(bars, success_rates):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{success_rate:.1%}', ha='center', va='bottom')
    
    # 学习曲线（如果有的话）
    for i, (name, results) in enumerate(agent_results.items()):
        episode_rewards = results['episode_rewards']
        # 计算滑动平均
        window_size = min(50, len(episode_rewards) // 4)
        if window_size > 1:
            smoothed_rewards = []
            for j in range(len(episode_rewards)):
                start_idx = max(0, j - window_size + 1)
                end_idx = j + 1
                smoothed_rewards.append(np.mean(episode_rewards[start_idx:end_idx]))
            axes[1, 1].plot(smoothed_rewards, label=name, color=colors[i % len(colors)])
    
    axes[1, 1].set_title('Episode Rewards (Smoothed)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比结果图表保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """评估执行入口"""
    parser = argparse.ArgumentParser(description='评估强化学习智能体')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--agent', type=str, choices=['dqn_rnd', 'dqn_icm'], 
                       required=True, help='智能体类型')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='评估局数')
    parser.add_argument('--render', action='store_true',
                       help='渲染环境')
    parser.add_argument('--output_dir', type=str, default='results/evaluations',
                       help='结果输出目录')
    parser.add_argument('--compare_models', type=str, nargs='+',
                       help='比较多个模型（提供多个模型路径）')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 加载配置
    config = Config(args.config)
    device = get_device(config)
    
    # 创建环境
    env = make_atari_env(args.env, config, args.seed)
    
    logger.info(f"评估环境: {args.env}")
    logger.info(f"使用设备: {device}")
    
    if args.compare_models:
        # 比较多个模型
        logger.info(f"比较 {len(args.compare_models)} 个模型")
        
        agent_results = {}
        
        for i, model_path in enumerate(args.compare_models):
            logger.info(f"评估模型 {i+1}/{len(args.compare_models)}: {model_path}")
            
            # 创建智能体并加载模型
            agent = create_agent(args.agent, config, device, env.action_space.n)
            agent.load(model_path)
            
            # 评估
            results = evaluate_agent(agent, env, args.num_episodes, args.render)
            
            model_name = os.path.basename(model_path).replace('.pth', '')
            agent_results[model_name] = results
            
            logger.info(f"模型 {model_name}: 平均奖励 = {results['mean_reward']:.2f}")
        
        # 绘制比较结果
        compare_path = os.path.join(args.output_dir, 'agent_comparison.png')
        compare_agents(agent_results, compare_path)
        
        # 保存详细结果
        import json
        results_path = os.path.join(args.output_dir, 'comparison_results.json')
        with open(results_path, 'w') as f:
            # 移除不能JSON序列化的内容
            serializable_results = {}
            for name, results in agent_results.items():
                serializable_results[name] = {
                    k: v for k, v in results.items() 
                    if k not in ['state_visits', 'episode_rewards', 'episode_intrinsic_rewards', 'episode_steps']
                }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"比较结果保存到: {args.output_dir}")
        
    else:
        # 评估单个模型
        logger.info(f"评估模型: {args.model_path}")
        
        # 创建智能体并加载模型
        agent = create_agent(args.agent, config, device, env.action_space.n)
        agent.load(args.model_path)
        
        logger.info(f"智能体加载完成: {type(agent).__name__}")
        
        # 评估
        results = evaluate_agent(agent, env, args.num_episodes, args.render)
        
        # 打印结果
        logger.info("="*50)
        logger.info("评估结果:")
        logger.info(f"平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        logger.info(f"最小/最大奖励: {results['min_reward']:.1f} / {results['max_reward']:.1f}")
        logger.info(f"中位数奖励: {results['median_reward']:.2f}")
        logger.info(f"成功率: {results['success_rate']:.1%}")
        logger.info(f"平均步数: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
        logger.info(f"平均内在奖励: {results['mean_intrinsic_reward']:.3f}")
        
        if 'exploration_efficiency' in results:
            logger.info(f"探索效率: {results['exploration_efficiency']:.3f}")
            logger.info(f"状态多样性: {results['state_diversity']:.3f}")
        
        logger.info("="*50)
        
        # 绘制结果
        plot_path = os.path.join(args.output_dir, 'evaluation_results.png')
        plot_evaluation_results(results, plot_path)
        
        # 保存详细结果
        import json
        results_path = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            # 移除不能JSON序列化的内容
            serializable_results = {
                k: v for k, v in results.items() 
                if k not in ['state_visits', 'episode_rewards', 'episode_intrinsic_rewards', 'episode_steps']
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"评估结果保存到: {args.output_dir}")
    
    env.close()


if __name__ == "__main__":
    main()
