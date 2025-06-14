# Montezuma's Revenge 强化学习解决方案

基于深度强化学习解决《蒙特祖玛的复仇》游戏中的稀疏奖励问题

## 项目概述

本项目使用探索型强化学习算法来解决Atari游戏《蒙特祖玛的复仇》中的稀疏奖励问题。游戏中奖励非常罕见，传统强化学习方法难以有效探索。本项目实现了两种主要算法：

- **DQN + RND (随机网络蒸馏)** - 使用预测误差作为内在奖励激励探索
- **DQN + ICM (内在好奇心模块)** - 基于状态转换预测误差驱动探索

这些算法通过生成额外的内在奖励信号，使智能体在稀少的外部奖励环境中依然能够进行有效探索。

## 快速入门

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

主要依赖：
- Python 3.8+
- PyTorch 1.9+
- Gymnasium[atari]
- NumPy, Matplotlib
- Tensorboard

### 2. 主要功能流程

1. **训练模型** - 使用内在奖励训练探索型智能体
2. **评估模型** - 测试智能体性能表现
3. **演示模型** - 可视化智能体游戏过程
4. **比较实验** - 对比不同算法性能

### 3. 命令行使用

```bash
# 1. 训练模型 (选择算法: dqn_rnd 或 dqn_icm)
python train.py --agent dqn_rnd

# 2. 评估模型性能
# (可使用训练过程中最佳模型或最终模型)
python evaluate.py --model_path saved_models/dqn_rnd_best.pth --agent dqn_rnd
# 或
python evaluate.py --model_path saved_models/dqn_rnd_final.pth --agent dqn_rnd

# 3. 演示模型游戏过程
python demo.py --model_path saved_models/dqn_rnd_best.pth --agent dqn_rnd

# 4. 比较不同算法性能
python compare_experiments.py --agents dqn_rnd dqn_icm

# 5. 实时监控训练过程
tensorboard --logdir logs/
```

## 模块功能详解

### 1. 训练模块 (train.py)

训练智能体探索游戏环境并学习最优策略。

```bash
python train.py [参数]
```

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--agent` | 智能体类型 (dqn_rnd 或 dqn_icm) | 必填 |
| `--config` | 配置文件路径 | config.yaml |
| `--log_dir` | 日志保存目录 | logs |
| `--model_dir` | 模型保存目录 | saved_models |
| `--resume` | 从检查点恢复训练 | 无 |

### 2. 评估模块 (evaluate.py)

评估已训练智能体在游戏中的性能表现。

```bash
python evaluate.py [参数]
```

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--model_path` | 模型文件路径 | 必填 |
| `--agent` | 智能体类型 | 必填 |
| `--num_episodes` | 评估局数 | 10 |
| `--render` | 是否渲染游戏画面 | 否 |

### 3. 演示模块 (demo.py)

直观展示智能体在游戏中的表现，支持保存录像。

```bash
python demo.py [参数]
```

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--model_path` | 模型文件路径 | 必填 |
| `--agent` | 智能体类型 | 必填 |
| `--episodes` | 演示局数 | 3 |
| `--save_gif` | 保存游戏回放为GIF | 否 |

### 4. 实验比较 (compare_experiments.py)

比较不同算法或参数设置的性能，生成对比图表。

```bash
python compare_experiments.py [参数]
```

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--agents` | 要比较的智能体类型列表 | dqn_rnd dqn_icm |
| `--runs` | 每个实验的重复次数 | 3 |
| `--load_only` | 只加载已有实验结果 | 否 |
| `--plot` | 生成对比图表 | 否 |

### 5. 配置系统 (config.yaml)

项目使用YAML格式的配置文件来管理环境和算法参数。主要配置项包括：

```yaml
# 环境配置
environment:
  name: "MontezumaRevengeNoFrameskip-v4"  # 游戏环境名称
  frame_stack: 4                           # 堆叠的帧数
  frame_skip: 4                            # 跳过的帧数

# DQN配置
dqn:
  learning_rate: 0.0001                    # 学习率
  gamma: 0.99                              # 折扣因子
  eps_start: 1.0                           # 初始探索率
  eps_end: 0.01                            # 最终探索率

# RND配置
rnd:
  feature_size: 512                        # 特征大小
  intrinsic_reward_weight: 0.1             # 内在奖励权重

# ICM配置
icm:
  feature_dim: 256                         # 特征维度
  forward_weight: 0.2                      # 前向模型权重
```

可以通过修改配置文件调整算法参数，无需更改代码。

## 项目结构

```
zuma/
├── agents/                # 智能体实现
│   ├── base_agent.py      # 基础智能体类
│   ├── dqn_rnd.py         # DQN+RND算法实现
│   └── dqn_icm.py         # DQN+ICM算法实现
├── models/                # 神经网络模型
│   ├── networks.py        # DQN网络结构
│   ├── rnd.py             # 随机网络蒸馏模型
│   └── icm.py             # 内在好奇心模型
├── utils/                 # 工具函数
│   ├── config.py          # 配置管理
│   ├── environment.py     # 环境包装
│   ├── replay_buffer.py   # 经验回放
│   └── visualization.py   # 可视化工具
├── logs/                  # 训练日志和TensorBoard数据
├── results/               # 评估结果和可视化图表
├── saved_models/          # 保存训练好的模型权重
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── demo.py                # 演示脚本
├── compare_experiments.py # 实验比较脚本
├── config.yaml            # 配置文件(环境和算法参数)
└── requirements.txt       # 依赖列表
```
