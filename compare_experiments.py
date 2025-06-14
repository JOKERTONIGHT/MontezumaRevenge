"""
实验比较脚本
比较DQN+RND和DQN+ICM的性能
"""

import os
import json
import argparse
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import yaml

from utils import Config

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, base_config_path: str, output_dir: str):
        self.base_config = Config(base_config_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = logging.getLogger("ExperimentRunner")
    
    def run_experiment(self, agent_type: str, config_modifications: Dict = None, 
                      experiment_name: str = None, num_runs: int = 3) -> List[Dict]:
        """运行实验"""
        if experiment_name is None:
            experiment_name = f"{agent_type}_{int(time.time())}"
        
        self.logger.info(f"开始实验: {experiment_name}")
        results = []
        
        experiment_dir = os.path.join(self.output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        for run in range(num_runs):
            run_name = f"run_{run+1}"
            run_dir = os.path.join(experiment_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            
            self.logger.info(f"运行 {run + 1}/{num_runs}")
            
            # 构建命令行
            cmd = [
                "python", "train.py",
                "--agent", agent_type,
                "--log_dir", os.path.join(run_dir, "logs"),
                "--model_dir", os.path.join(run_dir, "models")
            ]
            
            # 构建配置
            run_config = self.base_config.copy()
            if config_modifications:
                run_config.update(config_modifications)
            
            # 保存配置
            config_path = os.path.join(run_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(run_config.to_dict(), f)
            
            cmd.extend(["--config", config_path])
            
            # 运行训练
            import subprocess
            self.logger.info(f"执行命令: {' '.join(cmd)}")
            process = subprocess.run(cmd)
            
            if process.returncode != 0:
                self.logger.error(f"训练失败，退出码: {process.returncode}")
            
            # 评估模型
            eval_cmd = [
                "python", "evaluate.py",
                "--agent", agent_type,
                "--model_path", os.path.join(run_dir, "models", f"{agent_type}_final.pth"),
                "--config", config_path,
                "--output_dir", os.path.join(run_dir, "evaluation"),
                "--num_episodes", "20"
            ]
            
            self.logger.info(f"评估模型: {' '.join(eval_cmd)}")
            eval_process = subprocess.run(eval_cmd)
            
            # 收集结果
            try:
                with open(os.path.join(run_dir, "evaluation", "evaluation_results.json"), 'r') as f:
                    run_results = json.load(f)
                    run_results['run'] = run + 1
                    results.append(run_results)
            except:
                self.logger.error(f"无法读取评估结果")
        
        # 保存汇总结果
        summary_path = os.path.join(experiment_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"实验 {experiment_name} 完成，结果保存到 {summary_path}")
        return results


class ExperimentAnalyzer:
    """实验分析器"""
    
    def __init__(self, experiments_dir: str):
        self.experiments_dir = experiments_dir
        self.logger = logging.getLogger("ExperimentAnalyzer")
    
    def load_experiment_results(self, experiment_name: str) -> List[Dict]:
        """加载实验结果"""
        summary_path = os.path.join(self.experiments_dir, experiment_name, "summary.json")
        
        try:
            with open(summary_path, 'r') as f:
                return json.load(f)
        except:
            self.logger.error(f"无法加载实验结果: {summary_path}")
            return []
    
    def compare_experiments(self, experiment_names: List[str], save_path: str = None):
        """比较多个实验"""
        if not experiment_names:
            self.logger.error("未指定实验名称")
            return
        
        # 加载结果
        experiment_results = {}
        for name in experiment_names:
            results = self.load_experiment_results(name)
            if results:
                experiment_results[name] = results
        
        if not experiment_results:
            self.logger.error("没有找到有效的实验结果")
            return
        
        # 提取指标
        experiment_rewards = {}
        experiment_steps = {}
        
        for name, results in experiment_results.items():
            rewards = [r.get('mean_reward', 0) for r in results]
            steps = [r.get('mean_steps', 0) for r in results]
            experiment_rewards[name] = rewards
            experiment_steps[name] = steps
        
        # 打印统计信息
        self.logger.info("\n=== 实验比较结果 ===")
        
        for name, rewards in experiment_rewards.items():
            self.logger.info(f"\n{name}:")
            self.logger.info(f"  平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            self.logger.info(f"  奖励范围: {np.min(rewards):.2f} - {np.max(rewards):.2f}")
            self.logger.info(f"  平均步数: {np.mean(experiment_steps[name]):.1f}")
        
        # 绘制比较图表
        self._plot_comparison(experiment_rewards, experiment_steps, save_path)
    
    def _plot_comparison(self, experiment_rewards: Dict[str, List[float]], 
                       experiment_steps: Dict[str, List[float]], save_path: str = None):
        """绘制实验比较图表"""
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 10))
        
        # 设置颜色
        colors = ['#2C7BB6', '#D7191C', '#33A02C', '#FF7F00', '#984EA3']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 平均奖励比较
        reward_means = [np.mean(rewards) for rewards in experiment_rewards.values()]
        reward_stds = [np.std(rewards) for rewards in experiment_rewards.values()]
        
        axes[0, 0].bar(experiment_rewards.keys(), reward_means, yerr=reward_stds, 
                      capsize=10, color=colors[:len(experiment_rewards)])
        axes[0, 0].set_title('平均奖励比较', fontsize=14)
        axes[0, 0].set_ylabel('平均奖励', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 平均步数比较
        step_means = [np.mean(steps) for steps in experiment_steps.values()]
        step_stds = [np.std(steps) for steps in experiment_steps.values()]
        
        axes[0, 1].bar(experiment_steps.keys(), step_means, yerr=step_stds,
                     capsize=10, color=colors[:len(experiment_steps)])
        axes[0, 1].set_title('平均步数比较', fontsize=14)
        axes[0, 1].set_ylabel('平均步数', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 奖励分布
        for i, (name, rewards) in enumerate(experiment_rewards.items()):
            sns.kdeplot(rewards, ax=axes[1, 0], label=name, color=colors[i % len(colors)])
        axes[1, 0].set_title('奖励分布', fontsize=14)
        axes[1, 0].set_xlabel('奖励', fontsize=12)
        axes[1, 0].set_ylabel('密度', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 滑动平均奖励曲线
        window_size = 3  # 滑动窗口大小
        for i, (name, episode_rewards) in enumerate(experiment_rewards.items()):
            smoothed_rewards = []
            for j in range(len(episode_rewards)):
                start_idx = max(0, j - window_size + 1)
                end_idx = j + 1
                smoothed_rewards.append(np.mean(episode_rewards[start_idx:end_idx]))
            axes[1, 1].plot(smoothed_rewards, label=name, color=colors[i % len(colors)])
            
        axes[1, 1].set_title('奖励趋势', fontsize=14)
        axes[1, 1].set_xlabel('运行序号', fontsize=12)
        axes[1, 1].set_ylabel('平滑奖励', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"比较图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """实验比较入口"""
    parser = argparse.ArgumentParser(description='比较不同强化学习算法的性能')
    parser.add_argument('--agents', nargs='+', default=['dqn_rnd', 'dqn_icm'],
                       help='要比较的智能体类型列表')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='results/experiments',
                       help='实验结果输出目录')
    parser.add_argument('--runs', type=int, default=3,
                       help='每个实验的重复次数')
    parser.add_argument('--load_only', action='store_true',
                       help='只加载已有实验结果')
    parser.add_argument('--plot', action='store_true',
                       help='绘制对比图表')
    
    args = parser.parse_args()
    
    # 创建实验目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建实验运行器
    runner = ExperimentRunner(args.config, args.output_dir)
    analyzer = ExperimentAnalyzer(args.output_dir)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Experiment")
    
    # 如果不是仅加载模式，则运行实验
    if not args.load_only:
        logger.info("开始比较实验...")
        
        for agent_type in args.agents:
            logger.info(f"运行 {agent_type} 实验 ({args.runs} 次)...")
            runner.run_experiment(
                agent_type,
                {},  # 使用默认配置
                f"{agent_type}_experiment",
                args.runs
            )
        
        logger.info("所有实验完成!")
    
    # 分析并绘图
    logger.info("分析实验结果...")
    comparison_path = os.path.join(args.output_dir, 'experiment_comparison.png')
    
    # 获取所有实验名称
    experiment_dirs = [f"{agent}_experiment" for agent in args.agents]
    analyzer.compare_experiments(experiment_dirs, comparison_path)
    
    logger.info(f"比较结果保存到: {comparison_path}")


if __name__ == "__main__":
    main()
