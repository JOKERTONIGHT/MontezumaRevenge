"""
演示脚本
快速演示训练好的智能体玩蒙特祖玛的复仇
"""

import argparse
import time
import numpy as np
import torch
import cv2
import os

from utils import Config, set_random_seed, get_device, make_atari_env
from agents import create_agent


def demo_agent(agent, env, num_episodes: int = 3, render: bool = True, 
               save_gif: bool = False, gif_path: str = None):
    """演示智能体游戏过程"""
    
    frames = []
    
    for episode in range(num_episodes):
        print(f"\n=== 第 {episode + 1} 局游戏 ===")
        
        state, _ = env.reset()
        episode_reward = 0
        episode_intrinsic_reward = 0
        episode_steps = 0
        episode_frames = []
        done = False
        
        while not done:
            # 选择动作（贪心策略）
            action = agent.select_action(state, eval_mode=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 计算内在奖励
            intrinsic_reward = agent.compute_intrinsic_reward(state, action, next_state)
            
            episode_reward += reward
            episode_intrinsic_reward += intrinsic_reward
            episode_steps += 1
            
            if render:
                # 渲染环境
                frame = env.render()
                if save_gif and frame is not None:
                    episode_frames.append(frame)
                
                # 稍微减慢游戏速度以便观察
                time.sleep(0.05)
            
            # 打印有趣的信息
            if reward > 0:
                print(f"  步骤 {episode_steps}: 获得奖励 {reward}!")
            
            if episode_steps % 1000 == 0:
                print(f"  已执行 {episode_steps} 步，当前奖励: {episode_reward}")
            
            state = next_state
        
        print(f"游戏结束!")
        print(f"总奖励: {episode_reward}")
        print(f"内在奖励: {episode_intrinsic_reward:.3f}")
        print(f"游戏步数: {episode_steps}")
        
        if save_gif and episode_frames:
            frames.extend(episode_frames)
    
    # 保存GIF
    if save_gif and frames and gif_path:
        print(f"\n保存游戏录像到: {gif_path}")
        save_frames_as_gif(frames, gif_path)


def save_frames_as_gif(frames, path: str, fps: int = 30):
    """将帧序列保存为GIF"""
    try:
        import imageio
        
        # 调整帧大小以减少文件大小
        resized_frames = []
        for frame in frames[::2]:  # 跳帧以减少文件大小
            if len(frame.shape) == 3:
                # 调整大小
                resized_frame = cv2.resize(frame, (160, 160))
                resized_frames.append(resized_frame)
        
        # 保存GIF
        imageio.mimsave(path, resized_frames, fps=fps)
        print(f"GIF已保存: {path}")
        
    except ImportError:
        print("需要安装 imageio 来保存GIF: pip install imageio")
    except Exception as e:
        print(f"保存GIF时出错: {e}")


def interactive_demo(agent, env):
    """交互式演示"""
    print("\n=== 交互式演示模式 ===")
    print("按 Enter 继续下一步，输入 'q' 退出，输入 'a' 自动播放")
    
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    auto_mode = False
    
    while True:
        # 显示当前状态信息
        print(f"\n步骤 {episode_steps}, 当前奖励: {episode_reward}")
        
        # 选择动作
        action = agent.select_action(state, eval_mode=True)
        print(f"智能体选择动作: {action}")
        
        # 渲染环境
        env.render()
        
        if not auto_mode:
            user_input = input("按 Enter 继续, 'q' 退出, 'a' 自动播放: ").strip().lower()
            if user_input == 'q':
                break
            elif user_input == 'a':
                auto_mode = True
                print("切换到自动播放模式...")
        else:
            time.sleep(0.1)  # 自动模式下的延迟
        
        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_steps += 1
        
        if reward > 0:
            print(f"获得奖励: {reward}!")
        
        if done:
            print(f"\n游戏结束! 总奖励: {episode_reward}, 总步数: {episode_steps}")
            
            restart = input("重新开始? (y/n): ").strip().lower()
            if restart == 'y':
                state, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                auto_mode = False
            else:
                break
        else:
            state = next_state


def analyze_agent_behavior(agent, env, num_steps: int = 1000):
    """分析智能体行为"""
    print(f"\n=== 智能体行为分析 ({num_steps} 步) ===")
    
    state, _ = env.reset()
    action_counts = np.zeros(env.action_space.n)
    rewards_collected = []
    intrinsic_rewards = []
    
    for step in range(num_steps):
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 计算内在奖励
        intrinsic_reward = agent.compute_intrinsic_reward(state, action, next_state)
        
        action_counts[action] += 1
        if reward != 0:
            rewards_collected.append(reward)
        intrinsic_rewards.append(intrinsic_reward)
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    # 分析结果
    print("动作分布:")
    action_meanings = env.unwrapped.get_action_meanings()
    for i, (count, meaning) in enumerate(zip(action_counts, action_meanings)):
        percentage = count / num_steps * 100
        print(f"  {i:2d} - {meaning:15s}: {count:4.0f} ({percentage:5.1f}%)")
    
    print(f"\n奖励统计:")
    if rewards_collected:
        print(f"  收集到的非零奖励数: {len(rewards_collected)}")
        print(f"  平均奖励: {np.mean(rewards_collected):.2f}")
        print(f"  奖励总和: {np.sum(rewards_collected):.2f}")
    else:
        print("  未收集到任何奖励")
    
    print(f"\n内在奖励统计:")
    print(f"  平均内在奖励: {np.mean(intrinsic_rewards):.4f}")
    print(f"  内在奖励标准差: {np.std(intrinsic_rewards):.4f}")
    print(f"  最大内在奖励: {np.max(intrinsic_rewards):.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='演示强化学习智能体')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--agent', type=str, choices=['dqn_rnd', 'dqn_icm'], 
                       required=True, help='智能体类型')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--env', type=str, default='MontezumaRevengeNoFrameskip-v4',
                       help='环境名称')
    parser.add_argument('--episodes', type=int, default=3,
                       help='演示局数')
    parser.add_argument('--no-render', action='store_true',
                       help='不渲染画面')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式演示模式')
    parser.add_argument('--analyze', action='store_true',
                       help='分析智能体行为')
    parser.add_argument('--save_gif', action='store_true',
                       help='保存游戏录像为GIF')
    parser.add_argument('--gif_path', type=str, default='demo.gif',
                       help='GIF保存路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 加载配置
    config = Config(args.config)
    device = get_device(config)
    
    print(f"使用设备: {device}")
    print(f"智能体类型: {args.agent}")
    print(f"模型路径: {args.model_path}")
    
    # 创建环境
    env = make_atari_env(args.env, config, args.seed)
    print(f"环境: {args.env}")
    
    # 创建智能体并加载模型
    print("加载智能体...")
    agent = create_agent(args.agent, config, device, env.action_space.n)
    agent.load(args.model_path)
    print(f"智能体加载完成: {type(agent).__name__}")
    
    try:
        if args.analyze:
            # 行为分析模式
            analyze_agent_behavior(agent, env)
        
        elif args.interactive:
            # 交互式演示模式
            interactive_demo(agent, env)
        
        else:
            # 标准演示模式
            print(f"\n开始演示 {args.episodes} 局游戏...")
            demo_agent(agent, env, args.episodes, 
                      render=not args.no_render,
                      save_gif=args.save_gif, 
                      gif_path=args.gif_path if args.save_gif else None)
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    
    finally:
        env.close()
        print("\n演示结束")


if __name__ == "__main__":
    main()
