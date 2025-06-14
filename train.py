"""
训练脚本
蒙特祖玛的复仇强化学习训练主程序
"""

import os
import logging
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import (
    Config, parse_args, merge_args_with_config, set_random_seed, 
    get_device, make_atari_env, ReplayBuffer, PrioritizedReplayBuffer,
    TrainingVisualizer, create_training_report
)
from agents import create_agent


def setup_logging(log_dir: str):
    """设置训练日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def create_replay_buffer(config: Config, device: torch.device, state_shape: tuple):
    """创建经验回放缓冲区"""
    use_prioritized = config.get('dqn.use_prioritized_replay', False)
    capacity = config.get('dqn.memory_size', 1000000)
    
    if use_prioritized:
        alpha = config.get('dqn.priority_alpha', 0.6)
        beta = config.get('dqn.priority_beta_start', 0.4)
        beta_increment = config.get('dqn.priority_beta_increment', 0.001)
        return PrioritizedReplayBuffer(capacity, state_shape, device, alpha, beta, beta_increment)
    else:
        return ReplayBuffer(capacity, state_shape, device)


def train_agent(config: Config, args):
    """训练智能体"""
    # 设置随机种子
    seed = config.get('experiment.seed', 42)
    set_random_seed(seed)
    
    # 设备设置
    device = get_device(config)
    
    # 设置日志目录和记录器
    log_dir = os.path.join(args.log_dir, f"{args.agent}_{int(time.time())}")
    logger = setup_logging(log_dir)
    
    logger.info(f"开始训练 {args.agent} 智能体")
    
    # 创建环境
    env_name = config.get('environment.name', 'MontezumaRevengeNoFrameskip-v4')
    env = make_atari_env(env_name, config, seed)
    eval_env = make_atari_env(env_name, config, seed + 1)
    
    # 创建智能体
    agent = create_agent(args.agent, config, device, env.action_space.n)
    
    # 创建经验回放缓冲区
    replay_buffer = create_replay_buffer(config, device, env.observation_space.shape)
    
    # 创建可视化工具
    visualizer = TrainingVisualizer(os.path.join(log_dir, 'plots'))
    tb_writer = SummaryWriter(os.path.join(log_dir, 'tensorboard')) if config.get('logging.tensorboard', True) else None
    
    # 训练参数
    total_timesteps = config.get('training.total_timesteps', 50000000)
    eval_frequency = config.get('training.eval_frequency', 100000)
    eval_episodes = config.get('training.eval_episodes', 10)
    save_frequency = config.get('training.save_frequency', 500000)
    log_frequency = config.get('training.log_frequency', 10000)
    render_eval = config.get('training.render_eval', False)
    
    # 恢复训练
    start_step = 0
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        agent.load(args.resume)
        start_step = agent.steps_done
    
    # 训练循环
    logger.info("开始训练循环")
    
    episode = 0
    step = start_step
    state, _ = env.reset()
    episode_reward = 0
    episode_intrinsic_reward = 0
    episode_steps = 0
    episode_start_time = time.time()
    
    best_eval_reward = float('-inf')
    
    with tqdm(total=total_timesteps, initial=start_step, desc="训练进度") as pbar:
        while step < total_timesteps:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 计算内在奖励
            intrinsic_reward = agent.compute_intrinsic_reward(state, action, next_state)
            
            # 存储经验
            replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新统计
            episode_reward += reward
            episode_intrinsic_reward += intrinsic_reward
            episode_steps += 1
            step += 1
            
            # 训练智能体
            if step >= config.get('dqn.learning_starts', 50000):
                train_metrics = agent.step(replay_buffer, step)
                
                # 记录训练指标
                if train_metrics and step % log_frequency == 0:
                    visualizer.log_training_metrics(step, **train_metrics)
                    
                    if tb_writer:
                        for key, value in train_metrics.items():
                            tb_writer.add_scalar(f'train/{key}', value, step)
            
            # 处理episode结束
            if done:
                episode_time = time.time() - episode_start_time
                
                # 记录episode数据
                visualizer.log_episode(
                    episode=episode,
                    reward=episode_reward,
                    steps=episode_steps,
                    epsilon=agent.epsilon,
                    intrinsic_reward=episode_intrinsic_reward,
                    episode_time=episode_time
                )
                
                if tb_writer:
                    tb_writer.add_scalar('episode/reward', episode_reward, episode)
                    tb_writer.add_scalar('episode/intrinsic_reward', episode_intrinsic_reward, episode)
                    tb_writer.add_scalar('episode/steps', episode_steps, episode)
                    tb_writer.add_scalar('episode/epsilon', agent.epsilon, episode)
                
                # 自适应探索（如果支持）
                if hasattr(agent, 'adapt_exploration'):
                    agent.adapt_exploration(episode_reward)
                
                # 重置环境
                state, _ = env.reset()
                episode += 1
                episode_reward = 0
                episode_intrinsic_reward = 0
                episode_steps = 0
                episode_start_time = time.time()
                
                # 重置episode统计
                if hasattr(agent, 'reset_episode_stats'):
                    agent.reset_episode_stats()
            else:
                state = next_state
            
            # 评估
            if step % eval_frequency == 0 and step > 0:
                logger.info(f"步骤 {step}: 开始评估")
                
                if hasattr(agent, 'evaluate_with_intrinsic_rewards'):
                    eval_metrics = agent.evaluate_with_intrinsic_rewards(eval_env, eval_episodes)
                else:
                    eval_metrics = agent.evaluate(eval_env, eval_episodes, render_eval)
                
                logger.info(f"评估结果: 平均奖励 = {eval_metrics['eval_mean_reward']:.2f}")
                
                if tb_writer:
                    for key, value in eval_metrics.items():
                        tb_writer.add_scalar(f'eval/{key}', value, step)
                
                # 保存最佳模型
                if eval_metrics['eval_mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['eval_mean_reward']
                    best_model_path = os.path.join(args.model_dir, f"{args.agent}_best.pth")
                    agent.save(best_model_path)
                    logger.info(f"保存最佳模型: {best_model_path}")
            
            # 保存检查点
            if step % save_frequency == 0 and step > 0:
                checkpoint_path = os.path.join(args.model_dir, f"{args.agent}_step_{step}.pth")
                agent.save(checkpoint_path)
                logger.info(f"保存检查点: {checkpoint_path}")
            
            # 更新进度条
            pbar.update(1)
            if step % 10000 == 0:
                pbar.set_postfix({
                    'Episode': episode,
                    'Reward': f"{episode_reward:.1f}",
                    'Epsilon': f"{agent.epsilon:.3f}"
                })
    
    # 训练完成
    logger.info("训练完成")
    
    # 保存最终模型
    final_model_path = os.path.join(args.model_dir, f"{args.agent}_final.pth")
    agent.save(final_model_path)
    logger.info(f"保存最终模型: {final_model_path}")
    
    # 最终评估
    logger.info("进行最终评估")
    if hasattr(agent, 'evaluate_with_intrinsic_rewards'):
        final_eval_metrics = agent.evaluate_with_intrinsic_rewards(eval_env, eval_episodes * 2)
    else:
        final_eval_metrics = agent.evaluate(eval_env, eval_episodes * 2, render_eval)
    
    logger.info(f"最终评估结果: {final_eval_metrics}")
    
    # 生成可视化
    logger.info("生成训练可视化")
    visualizer.plot_training_progress()
    visualizer.plot_loss_curves()
    visualizer.plot_reward_distribution()
    visualizer.save_data()
    
    # 生成训练报告
    create_training_report(visualizer, config.to_dict(), final_eval_metrics)
    
    # 关闭资源
    if tb_writer:
        tb_writer.close()
    env.close()
    eval_env.close()
    
    logger.info(f"训练结果保存在: {log_dir}")
    return final_eval_metrics


def main():
    """训练执行入口"""
    # 解析命令行参数并加载配置
    args = parse_args()
    config = Config(args.config)
    config = merge_args_with_config(args, config)
    
    # 创建必要目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 执行训练
    try:
        final_metrics = train_agent(config, args)
        print(f"\n训练完成! 最终平均奖励: {final_metrics.get('eval_mean_reward', 0):.2f}")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
