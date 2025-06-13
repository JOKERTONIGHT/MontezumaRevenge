"""
智能体模块
提供DQN+RND和DQN+ICM等强化学习智能体实现
"""

from .base_agent import BaseAgent, PrioritizedAgent
from .dqn_rnd import DQNRNDAgent
from .dqn_icm import DQNICMAgent

__all__ = [
    'BaseAgent',
    'PrioritizedAgent',
    'DQNRNDAgent',
    'DQNICMAgent'
]


def create_agent(agent_type: str, config, device, num_actions: int = 18):
    """智能体工厂函数"""
    if agent_type == 'dqn_rnd':
        return DQNRNDAgent(config, device, num_actions)
    elif agent_type == 'dqn_icm':
        return DQNICMAgent(config, device, num_actions)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
