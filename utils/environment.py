"""
环境包装器模块
提供Atari环境的预处理和包装功能
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import torch


class NoOpResetEnv(gym.Wrapper):
    """在重置时执行随机数量的无操作动作"""
    
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """如果游戏需要FIRE动作开始，则在重置时执行"""
    
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """将生命损失作为一局结束"""
    
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """跳帧并取最大值"""
    
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    """将奖励裁剪到[-1, 1]"""
    
    def reward(self, reward):
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """将帧大小调整为84x84并转换为灰度"""
    
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            obs = self._convert_obs(obs)
        else:
            obs[self._key] = self._convert_obs(obs[self._key])
        return obs
    
    def _convert_obs(self, obs):
        if self._grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            obs = np.expand_dims(obs, -1)
        return obs


class FrameStack(gym.Wrapper):
    """堆叠最后k帧"""
    
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=-1)


class ScaledFloatFrame(gym.ObservationWrapper):
    """将观察值标准化到[0, 1]"""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class TransposeFrame(gym.ObservationWrapper):
    """将观察值从(H, W, C)转换为(C, H, W)格式"""
    
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        
        # 检查是否已经是(C, H, W)格式
        if len(obs_shape) == 3 and obs_shape[0] <= 4 and obs_shape[1] == obs_shape[2]:
            # 已经是(C, H, W)格式，不需要转置
            self.observation_space = env.observation_space
            self.need_transpose = False
        else:
            # 需要从(H, W, C)转换为(C, H, W)
            self.observation_space = gym.spaces.Box(
                low=0, high=1, 
                shape=(obs_shape[2], obs_shape[0], obs_shape[1]), 
                dtype=env.observation_space.dtype
            )
            self.need_transpose = True

    def observation(self, observation):
        if self.need_transpose:
            return np.transpose(observation, (2, 0, 1))
        else:
            return observation


def make_atari_env(env_name, config, seed=None):
    """创建Atari环境"""
    # 确保导入ale_py以注册Atari环境
    import ale_py
    
    env = gym.make(env_name)
    
    if seed is not None:
        env.reset(seed=seed)
    
    # 应用包装器
    env = NoOpResetEnv(env, noop_max=config.get('environment.no_op_max', 30))
    
    if config.get('environment.fire_first', True):
        env = FireResetEnv(env)
    
    env = MaxAndSkipEnv(env, skip=config.get('environment.frame_skip', 4))
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, config.get('environment.frame_stack', 4))
    env = ScaledFloatFrame(env)
    env = TransposeFrame(env)  # 转换为(C, H, W)格式
    env = TransposeFrame(env)
    
    # 设置最大步数
    max_steps = config.get('environment.max_episode_steps', 18000)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    
    return env


class StateNormalizer:
    """状态标准化器"""
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
        self.count = 0
    
    def update(self, x):
        """更新统计信息"""
        if self.running_mean is None:
            self.running_mean = np.zeros_like(x)
            self.running_var = np.zeros_like(x)
        
        self.count += 1
        delta = x - self.running_mean
        self.running_mean += delta / self.count
        delta2 = x - self.running_mean
        self.running_var += delta * delta2
    
    def normalize(self, x):
        """标准化输入"""
        if self.running_mean is None:
            return x
        
        mean = self.running_mean
        var = self.running_var / max(1, self.count - 1)
        std = np.sqrt(var + self.epsilon)
        return (x - mean) / std
    
    def __call__(self, x, update_stats=True):
        if update_stats:
            self.update(x)
        return self.normalize(x)
