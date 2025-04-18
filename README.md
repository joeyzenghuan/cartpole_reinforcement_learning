# CartPole强化学习优化项目

这个项目使用Q-learning算法解决经典的CartPole平衡问题，并包含超参数优化工具，帮助找到最佳训练参数。

## 文件说明

- `cartpole_rl_tutorial.py` - 基础的CartPole教程，使用Q-learning算法训练智能体保持杆子平衡
- `cartpole_hyperparameter_optimizer.py` - 完整的超参数网格搜索优化工具，需要pandas库
- `cartpole_optimal_params.py` - 简化版超参数优化工具，快速测试关键超参数组合
- `optimal_cartpole.py` - 使用已优化的最佳超参数训练CartPole，包含保存/加载模型功能

## 环境要求

- Python 3.7+
- Gymnasium库（新版本的Gym）
- NumPy
- Matplotlib
- Pandas（仅`cartpole_hyperparameter_optimizer.py`需要）

安装依赖：

```bash
pip install gymnasium numpy matplotlib
pip install pandas  # 如果需要运行完整的参数优化工具
```

## 使用说明

### 1. 基础教程

运行基础的CartPole Q-learning训练：

```bash
python cartpole_rl_tutorial.py
```

这个脚本会使用默认参数训练一个智能体，并在训练结束后显示训练统计和Q值分析。

### 2. 超参数优化

运行简化版超参数优化（推荐）：

```bash
python cartpole_optimal_params.py
```

这个脚本会测试12种不同的超参数组合，找出性能最好的参数设置。运行时间约10-15分钟。

如果需要进行更完整的网格搜索（测试更多参数组合）：

```bash
python cartpole_hyperparameter_optimizer.py
```

这个脚本会测试所有参数组合（共108种），运行时间较长（1-2小时）。

### 3. 使用最优参数训练

使用已优化的最佳参数训练CartPole：

```bash
python optimal_cartpole.py
```

这个脚本使用我们找到的最优超参数组合训练智能体，并在训练后保存Q表和学习曲线，最后在可视化环境中展示训练效果。

## 最优超参数

通过超参数优化，我们发现以下超参数组合表现最好：

- 状态空间离散化粒度(n_bins): 12
- 学习率(alpha): 0.1
- 折扣因子(gamma): 0.99
- 初始探索率(epsilon_start): 0.3
- 探索率衰减(epsilon_decay): 0.995
- 最小探索率(epsilon_end): 0.01

使用这些参数，智能体通常能够在200-300回合内稳定学习到保持杆子平衡的策略。

## 进阶学习方向

- 尝试深度Q网络(DQN)处理更复杂环境
- 探索基于策略的方法如策略梯度(Policy Gradient)
- 学习Actor-Critic架构结合价值和策略方法的优点

## 项目简介

这是一个面向强化学习初学者的入门项目，使用经典的CartPole平衡问题作为示例。项目使用简单的Q-learning算法训练一个智能体，学习如何通过左右移动小车来保持杆子平衡。

![CartPole环境](https://gymnasium.farama.org/_images/cart_pole.gif)

## 什么是CartPole问题？

CartPole（倒立摆）是强化学习中的经典控制问题：

- 一个小车可以在一维轨道上左右移动
- 小车上连接着一根竖直的杆子
- 目标是通过控制小车的移动来保持杆子平衡，不让它倒下
- 每保持平衡一个时间步，获得+1奖励
- 当杆子倾斜角度过大或小车移出边界时，游戏结束

## 什么是强化学习？

强化学习是机器学习的一个分支，通过智能体与环境的交互来学习最优策略：

1. 智能体观察环境状态
2. 执行一个动作
3. 获得奖励和新的状态
4. 通过这些经验逐步调整策略

本项目使用Q-learning算法，这是一种简单而强大的强化学习方法，通过构建状态-动作价值表（Q表）来学习最优策略。

## 运行环境

运行本项目需要以下Python库：

```
gymnasium
numpy
matplotlib
```

## 学习要点

通过这个项目，你将学习强化学习的核心概念：

- 状态和动作空间
- 探索与利用的平衡
- Q-learning算法
- 奖励和折扣因子
- 连续状态空间的离散化
- 策略评估和改进

## 参考资源

- [Gymnasium文档](https://gymnasium.farama.org/)
- [强化学习简介 - Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)
- [强化学习基础 - 莫烦Python](https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/) 