"""
配置管理模块
处理YAML配置文件的加载和参数管理
"""

import yaml
import argparse
import torch
import random
import numpy as np
from typing import Dict, Any


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = None):
        if config_path:
            self.load_from_file(config_path)
        else:
            self.config = {}
    
    def load_from_file(self, config_path: str):
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """获取配置值，支持嵌套键如'dqn.learning_rate'"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def update(self, updates: Dict):
        """更新配置"""
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return self.config.copy()


def set_random_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: Config) -> torch.device:
    """获取计算设备"""
    device_name = config.get('experiment.device', 'cuda')
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Montezuma Revenge RL Training')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--agent', type=str, choices=['dqn_rnd', 'dqn_icm'], 
                        default='dqn_rnd', help='选择智能体类型')
    parser.add_argument('--env', type=str, default='MontezumaRevengeNoFrameskip-v4',
                        help='环境名称')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                        default=None, help='计算设备')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--model_dir', type=str, default='saved_models',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的模型路径')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式')
    parser.add_argument('--render', action='store_true',
                        help='渲染环境')
    
    return parser.parse_args()


def merge_args_with_config(args, config: Config) -> Config:
    """将命令行参数合并到配置中"""
    updates = {}
    
    if args.agent:
        updates['experiment'] = updates.get('experiment', {})
        updates['experiment']['agent_type'] = args.agent
    
    if args.env:
        updates['environment'] = updates.get('environment', {})
        updates['environment']['name'] = args.env
    
    if args.seed is not None:
        updates['experiment'] = updates.get('experiment', {})
        updates['experiment']['seed'] = args.seed
    
    if args.device:
        updates['experiment'] = updates.get('experiment', {})
        updates['experiment']['device'] = args.device
    
    if args.render:
        updates['training'] = updates.get('training', {})
        updates['training']['render_eval'] = True
    
    config.update(updates)
    return config
