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
import subprocess
import yaml

from utils import Config, TrainingVisualizer


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, base_config_path: str, output_dir: str):
        self.base_config = Config(base_config_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ExperimentRunner")
    
    def run_experiment(self, agent_type: str, config_modifications: Dict = None, 
                      experiment_name: str = None, num_runs: int = 3) -> List[Dict]:
        """运行实验"""
        if experiment_name is None:
            experiment_name = f"{agent_type}_{int(time.time())}"
        
        self.logger.info(f"开始实验: {experiment_name}")
        
        results = []
        
        for run in range(num_runs):
            self.logger.info(f"运行 {run + 1}/{num_runs}")
            
            # 创建实验配置
            config = Config()
            config.config = self.base_config.config.copy()
            
            if config_modifications:
                config.update(config_modifications)
            
            # 设置随机种子
            seed = 42 + run * 1000
            config.set('experiment.seed', seed)
            
            # 保存配置
            experiment_dir = os.path.join(self.output_dir, experiment_name, f"run_{run}")
            os.makedirs(experiment_dir, exist_ok=True)
            
            config_path = os.path.join(experiment_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
            
            # 运行训练
            log_dir = os.path.join(experiment_dir, "logs")
            model_dir = os.path.join(experiment_dir, "models")
            
            cmd = [
                "python", "train.py",
                "--config", config_path,
                "--agent", agent_type,
                "--log_dir", log_dir,
                "--model_dir", model_dir,
                "--seed", str(seed)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600*8)  # 8小时超时
                
                if result.returncode == 0:
                    self.logger.info(f"运行 {run + 1} 完成")
                    
                    # 加载结果
                    results_path = os.path.join(log_dir, "training_data.npz")
                    if os.path.exists(results_path):
                        run_result = self.load_run_results(results_path)
                        run_result['run_id'] = run
                        run_result['experiment_name'] = experiment_name
                        run_result['agent_type'] = agent_type
                        results.append(run_result)
                    else:
                        self.logger.warning(f"运行 {run + 1} 结果文件未找到")
                else:
                    self.logger.error(f"运行 {run + 1} 失败: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"运行 {run + 1} 超时")
            except Exception as e:
                self.logger.error(f"运行 {run + 1} 出错: {e}")
        
        # 保存实验结果
        self.save_experiment_results(experiment_name, results)
        
        return results
    
    def load_run_results(self, results_path: str) -> Dict:
        """加载单次运行结果"""
        data = np.load(results_path)
        
        result = {}
        
        # 提取episode数据
        if 'episode_reward' in data:
            rewards = data['episode_reward']
            result['final_reward'] = float(rewards[-100:].mean()) if len(rewards) >= 100 else float(rewards.mean())
            result['max_reward'] = float(rewards.max())
            result['reward_std'] = float(rewards[-100:].std()) if len(rewards) >= 100 else float(rewards.std())
            result['total_episodes'] = len(rewards)
            
            # 学习曲线
            result['learning_curve'] = rewards.tolist()
        
        # 提取训练指标
        metrics = ['q_loss', 'rnd_loss', 'icm_loss', 'intrinsic_reward_mean']
        for metric in metrics:
            if f'metric_{metric}' in data:
                values = data[f'metric_{metric}']
                if len(values) > 0:
                    result[f'final_{metric}'] = float(values[-100:].mean()) if len(values) >= 100 else float(values.mean())
        
        return result
    
    def save_experiment_results(self, experiment_name: str, results: List[Dict]):
        """保存实验结果"""
        results_path = os.path.join(self.output_dir, experiment_name, "experiment_results.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"实验结果保存到: {results_path}")


class ExperimentAnalyzer:
    """实验分析器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_experiment_results(self, experiment_name: str) -> List[Dict]:
        """加载实验结果"""
        results_path = os.path.join(self.output_dir, experiment_name, "experiment_results.json")
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def compare_experiments(self, experiment_names: List[str], save_path: str = None):
        """比较多个实验"""
        all_results = {}
        
        for name in experiment_names:
            try:
                results = self.load_experiment_results(name)
                all_results[name] = results
            except FileNotFoundError:
                print(f"警告: 未找到实验 {name} 的结果文件")
        
        if not all_results:
            print("没有找到任何实验结果")
            return
        
        # 创建比较图表
        self.plot_experiment_comparison(all_results, save_path)
        
        # 统计分析
        self.statistical_analysis(all_results)
    
    def plot_experiment_comparison(self, all_results: Dict[str, List[Dict]], save_path: str = None):
        """绘制实验比较图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Experiment Comparison: DQN+RND vs DQN+ICM', fontsize=16)
        
        # 提取数据
        experiment_data = {}
        for exp_name, results in all_results.items():
            experiment_data[exp_name] = {
                'final_rewards': [r['final_reward'] for r in results if 'final_reward' in r],
                'max_rewards': [r['max_reward'] for r in results if 'max_reward' in r],
                'learning_curves': [r['learning_curve'] for r in results if 'learning_curve' in r]
            }
        
        # 1. 最终奖励比较
        exp_names = list(experiment_data.keys())
        final_rewards_data = [experiment_data[name]['final_rewards'] for name in exp_names]
        
        axes[0, 0].boxplot(final_rewards_data, labels=exp_names)
        axes[0, 0].set_title('Final Performance Comparison')
        axes[0, 0].set_ylabel('Final Average Reward (Last 100 Episodes)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加均值标记
        for i, rewards in enumerate(final_rewards_data):
            if rewards:
                mean_reward = np.mean(rewards)
                axes[0, 0].scatter(i + 1, mean_reward, color='red', s=50, zorder=5)
                axes[0, 0].text(i + 1, mean_reward + 5, f'{mean_reward:.1f}', 
                               ha='center', va='bottom', fontweight='bold')
        
        # 2. 最大奖励比较
        max_rewards_data = [experiment_data[name]['max_rewards'] for name in exp_names]
        
        axes[0, 1].boxplot(max_rewards_data, labels=exp_names)
        axes[0, 1].set_title('Maximum Reward Comparison')
        axes[0, 1].set_ylabel('Maximum Episode Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 学习曲线比较
        colors = ['blue', 'red', 'green', 'orange']
        for i, (exp_name, data) in enumerate(experiment_data.items()):
            learning_curves = data['learning_curves']
            if learning_curves:
                # 计算平均学习曲线
                min_length = min(len(curve) for curve in learning_curves)
                truncated_curves = [curve[:min_length] for curve in learning_curves]
                mean_curve = np.mean(truncated_curves, axis=0)
                std_curve = np.std(truncated_curves, axis=0)
                
                # 平滑曲线
                window_size = min(100, len(mean_curve) // 10)
                if window_size > 1:
                    smoothed_mean = np.convolve(mean_curve, np.ones(window_size)/window_size, mode='valid')
                    x = np.arange(len(smoothed_mean))
                    axes[1, 0].plot(x, smoothed_mean, label=exp_name, color=colors[i % len(colors)], linewidth=2)
        
        axes[1, 0].set_title('Learning Curves')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Reward (Smoothed)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 性能统计表
        stats_text = "Performance Statistics:\n\n"
        for exp_name, data in experiment_data.items():
            final_rewards = data['final_rewards']
            if final_rewards:
                mean_reward = np.mean(final_rewards)
                std_reward = np.std(final_rewards)
                max_reward = np.max(data['max_rewards']) if data['max_rewards'] else 0
                
                stats_text += f"{exp_name}:\n"
                stats_text += f"  Mean Final Reward: {mean_reward:.2f} ± {std_reward:.2f}\n"
                stats_text += f"  Best Episode Reward: {max_reward:.1f}\n"
                stats_text += f"  Number of Runs: {len(final_rewards)}\n\n"
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Statistics Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"比较图表保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def statistical_analysis(self, all_results: Dict[str, List[Dict]]):
        """统计显著性分析"""
        print("\n" + "="*50)
        print("统计分析结果")
        print("="*50)
        
        # 提取最终奖励数据
        experiment_rewards = {}
        for exp_name, results in all_results.items():
            rewards = [r['final_reward'] for r in results if 'final_reward' in r]
            experiment_rewards[exp_name] = rewards
        
        # 基础统计
        for exp_name, rewards in experiment_rewards.items():
            if rewards:
                print(f"\n{exp_name}:")
                print(f"  样本数: {len(rewards)}")
                print(f"  均值: {np.mean(rewards):.3f}")
                print(f"  标准差: {np.std(rewards):.3f}")
                print(f"  最小值: {np.min(rewards):.3f}")
                print(f"  最大值: {np.max(rewards):.3f}")
                print(f"  中位数: {np.median(rewards):.3f}")
        
        # 如果有两个实验，进行t检验
        if len(experiment_rewards) == 2:
            exp_names = list(experiment_rewards.keys())
            rewards1 = experiment_rewards[exp_names[0]]
            rewards2 = experiment_rewards[exp_names[1]]
            
            if len(rewards1) > 1 and len(rewards2) > 1:
                try:
                    from scipy import stats
                    
                    # 进行独立样本t检验
                    t_stat, p_value = stats.ttest_ind(rewards1, rewards2)
                    
                    print(f"\n独立样本t检验:")
                    print(f"  {exp_names[0]} vs {exp_names[1]}")
                    print(f"  t统计量: {t_stat:.4f}")
                    print(f"  p值: {p_value:.4f}")
                    
                    if p_value < 0.05:
                        better_exp = exp_names[0] if np.mean(rewards1) > np.mean(rewards2) else exp_names[1]
                        print(f"  结果: {better_exp} 显著优于另一个方法 (p < 0.05)")
                    else:
                        print(f"  结果: 两种方法之间没有显著差异 (p >= 0.05)")
                    
                    # 效应大小 (Cohen's d)
                    pooled_std = np.sqrt(((len(rewards1) - 1) * np.var(rewards1, ddof=1) + 
                                         (len(rewards2) - 1) * np.var(rewards2, ddof=1)) / 
                                        (len(rewards1) + len(rewards2) - 2))
                    cohens_d = (np.mean(rewards1) - np.mean(rewards2)) / pooled_std
                    print(f"  Cohen's d (效应大小): {cohens_d:.4f}")
                    
                    if abs(cohens_d) < 0.2:
                        effect_size = "小"
                    elif abs(cohens_d) < 0.8:
                        effect_size = "中等"
                    else:
                        effect_size = "大"
                    print(f"  效应大小解释: {effect_size}")
                
                except ImportError:
                    print("\n需要安装scipy进行统计检验: pip install scipy")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行和比较强化学习实验')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='基础配置文件路径')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='实验输出目录')
    parser.add_argument('--run_experiments', action='store_true',
                       help='运行新实验')
    parser.add_argument('--analyze_experiments', type=str, nargs='+',
                       help='分析指定的实验')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='每个实验的运行次数')
    parser.add_argument('--quick_test', action='store_true',
                       help='快速测试模式（减少训练步数）')
    
    args = parser.parse_args()
    
    if args.run_experiments:
        # 运行实验
        runner = ExperimentRunner(args.config, args.output_dir)
        
        # 实验配置
        base_modifications = {}
        if args.quick_test:
            base_modifications = {
                'training': {
                    'total_timesteps': 1000000,  # 减少到100万步
                    'eval_frequency': 50000,
                    'save_frequency': 200000
                }
            }
        
        # 运行DQN+RND实验
        print("运行 DQN+RND 实验...")
        rnd_results = runner.run_experiment(
            'dqn_rnd', 
            base_modifications,
            'dqn_rnd_experiment',
            args.num_runs
        )
        
        # 运行DQN+ICM实验
        print("运行 DQN+ICM 实验...")
        icm_results = runner.run_experiment(
            'dqn_icm',
            base_modifications,
            'dqn_icm_experiment', 
            args.num_runs
        )
        
        print("所有实验完成!")
    
    if args.analyze_experiments:
        # 分析实验
        analyzer = ExperimentAnalyzer(args.output_dir)
        
        print("分析实验结果...")
        comparison_path = os.path.join(args.output_dir, 'experiment_comparison.png')
        analyzer.compare_experiments(args.analyze_experiments, comparison_path)
        
        print("分析完成!")
    
    if not args.run_experiments and not args.analyze_experiments:
        print("请指定 --run_experiments 或 --analyze_experiments")
        print("例如:")
        print("  python compare_experiments.py --run_experiments --num_runs 3")
        print("  python compare_experiments.py --analyze_experiments dqn_rnd_experiment dqn_icm_experiment")


if __name__ == "__main__":
    main()
