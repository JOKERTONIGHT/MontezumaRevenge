# Montezuma's Revenge 强化学习解决方案

基于深度强化学习解决《蒙特祖玛的复仇》游戏中的稀疏奖励问题

## 项目概述

本项目实现了两种强化学习算法来提高Atari游戏《蒙特祖玛的复仇》中的探索效率：
- **DQN + RND** - 使用随机网络蒸馏生成内在奖励
- **DQN + ICM** - 基于预测误差的内在动机机制

## 核心特性

- 🎯 稀疏奖励环境的高效探索
- 🧠 基于好奇心的内在动机机制
- 📊 完整的训练监控与可视化
- 🔄 可复现的实验配置

## 算法原理

| 算法 | 核心机制 | 适用场景 |
|------|---------|---------|
| **DQN + RND** | 随机网络蒸馏生成新颖状态奖励 | 高度稀疏奖励环境 |
| **DQN + ICM** | 基于预测误差的内在好奇心 | 需要状态转换理解的环境 |

## 环境配置

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

## 使用指南

### 训练模型

```bash
# 训练DQN+RND模型
python train.py --agent dqn_rnd --env MontezumaRevengeNoFrameskip-v4

# 训练DQN+ICM模型
python train.py --agent dqn_icm --env MontezumaRevengeNoFrameskip-v4
```

### 评估与可视化

```bash
# 评估训练好的模型
python evaluate.py --model_path saved_models/dqn_rnd_best.pth

# 可视化训练过程
tensorboard --logdir logs/
```

### 演示游戏

使用`demo.py`脚本可以直观地演示训练好的智能体如何在游戏中表现：

```bash
# 演示DQN+RND模型游玩
python demo.py --model_path saved_models/dqn_rnd_best.pth --episodes 3

# 保存游戏回放为GIF
python demo.py --model_path saved_models/dqn_icm_best.pth --save_gif --gif_path results/demo.gif

# 不渲染界面（仅记录统计数据）
python demo.py --model_path saved_models/dqn_rnd_best.pth --no_render
```

### 实验比较

使用`compare_experiments.py`脚本进行不同算法的系统性比较实验：

```bash
# 比较DQN+RND和DQN+ICM性能（默认各进行3次运行）
python compare_experiments.py --agents dqn_rnd dqn_icm --runs 3

# 自定义实验名称并指定输出目录
python compare_experiments.py --agents dqn_rnd dqn_icm --experiment curiosity_comparison --output_dir results/comparisons

# 生成比较图表
python compare_experiments.py --load_only --experiment_dirs results/exp1 results/exp2 --plot
```

## 可视化训练过程详解

本项目提供了两种训练可视化方式：

### 1. TensorBoard 实时监控

训练过程中，关键指标会被记录到TensorBoard，包括：

- **奖励**：外部奖励、内部奖励和总奖励
- **损失函数**：DQN损失、RND/ICM预测损失
- **探索指标**：状态访问热图、epsilon值变化
- **网络梯度**：各层参数梯度范数

启动TensorBoard查看训练实时数据：

```bash
tensorboard --logdir logs/
```

访问`http://localhost:6006`查看可视化界面。

### 2. 训练报告生成

训练结束后，系统会自动生成包含以下内容的可视化报告：

- 奖励曲线（含滑动平均）
- 每轮步数统计
- 探索覆盖率分析
- 网络权重分布

训练报告保存在`results/`目录下。

### 可视化系统工作原理

训练过程中，`TrainingVisualizer`类负责：

1. 收集训练指标并保存到内存和磁盘
2. 定期生成可视化图表
3. 通过TensorBoard Writer记录实时数据

每个实验会生成独立的日志目录，便于比较不同算法和参数设置的性能差异。

## 项目结构

```
zuma/
├── agents/           # 智能体实现
│   ├── dqn_rnd.py    # DQN+RND算法
│   ├── dqn_icm.py    # DQN+ICM算法
│   └── base_agent.py # 基础智能体类
├── models/           # 神经网络模型
├── utils/            # 工具函数
├── logs/             # 训练日志
├── saved_models/     # 保存的模型
├── results/          # 实验结果
├── train.py          # 训练脚本
├── evaluate.py       # 评估脚本
├── demo.py           # 游戏演示脚本
├── compare_experiments.py # 实验比较脚本
├── config.yaml       # 配置文件
└── requirements.txt  # 依赖列表
```

## 实验结果

详细的实验结果和分析请查看 `results/` 目录。

## 贡献

欢迎提交Issue和Pull Request来改进项目。
