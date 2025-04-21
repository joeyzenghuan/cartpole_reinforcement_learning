"""
强化学习进阶教程：使用PPO算法训练CartPole平衡问题（修正版）
===========================================================

修复问题：
1. 价值损失计算维度不匹配问题
2. 优势函数和返回值计算逻辑优化

本教程详细介绍了近端策略优化算法(PPO)，这是一种先进的策略梯度强化学习方法。
适合强化学习初学者理解和实践。
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
import platform
import sys
import locale

# 检测系统是否支持中文输出
def check_chinese_support():
    # 获取系统默认编码
    system_encoding = locale.getpreferredencoding()
    
    # Windows下检查控制台代码页
    if platform.system() == "Windows":
        try:
            import ctypes
            console_cp = ctypes.windll.kernel32.GetConsoleOutputCP()
            # 检查是否为中文代码页(936是中文GBK，65001是UTF-8)
            return console_cp in (936, 65001)
        except:
            # 如果无法获取代码页，检查默认编码
            return system_encoding.lower() in ('gbk', 'cp936', 'utf-8', 'utf8')
    else:
        # 非Windows系统假设支持UTF-8
        return True

# 根据系统支持情况选择语言
USE_CHINESE = check_chinese_support()

# 配置matplotlib支持中文
if platform.system() == "Windows":
    # Windows系统使用微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
elif platform.system() == "Linux":
    # Linux系统尝试使用文泉驿微米黑
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
elif platform.system() == "Darwin":
    # macOS系统使用苹方
    plt.rcParams['font.sans-serif'] = ['PingFang SC']
    
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 超参数配置
class Config:
    # 环境参数
    env_name = "CartPole-v1"  # CartPole平衡杆环境，目标是保持杆子直立
    state_dim = 4       # 状态空间维度：小车位置、速度、杆子角度、角速度
    action_dim = 2      # 动作空间维度：向左/向右推小车
    
    # 训练参数
    learning_rate = 3e-4  # 学习率，控制网络参数更新步长
    gamma = 0.99        # 折扣因子，决定未来奖励的重要性
    gae_lambda = 0.95   # 广义优势估计(GAE)参数，平衡偏差和方差
    ppo_eps = 0.2       # PPO裁剪阈值，限制策略更新幅度
    epochs = 4          # 每次数据收集后的更新轮数
    batch_size = 64     # 最小批量大小，每次更新使用的样本数
    horizon = 2048      # 每次收集的经验步数
    
    # 网络结构
    hidden_dim = 64     # 隐藏层维度
    
    # 训练控制
    max_episodes = 200  # 最大训练回合数
    save_freq = 50      # 模型保存频率
    render = False      # 是否显示训练过程
    
    # 评估参数
    eval_episodes = 10  # 评估时执行的回合数

# PPO网络架构（Actor-Critic共享特征提取层）
class PPO(nn.Module):
    def __init__(self, cfg):
        """
        初始化PPO网络，使用Actor-Critic架构
        
        Actor：策略网络，输出动作概率分布
        Critic：价值网络，估计状态价值
        共享特征提取层提高了训练效率
        """
        super().__init__()
        # 共享的特征提取层 - 提取状态的关键特征
        self.feature = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),  # 输入层到隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),  # 第二个隐藏层
            nn.ReLU()   # 激活函数
        )
        
        # Actor网络（策略）- 决定在给定状态下采取什么动作
        self.actor = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        
        # Critic网络（价值估计）- 评估状态的价值
        self.critic = nn.Linear(cfg.hidden_dim, 1)
        
    def forward(self, x):
        """前向传播，同时计算动作概率和状态价值"""
        features = self.feature(x)
        return self.actor(features), self.critic(features)
    
    def act(self, state):
        """
        根据当前状态选择动作
        
        返回：
        - 选择的动作
        - 该动作的对数概率（用于后续PPO更新）
        - 当前状态的价值估计
        """
        with torch.no_grad():  # 不计算梯度，提高推理速度
            logits, value = self.forward(state)
        # 使用Categorical分布采样动作
        dist = Categorical(logits=logits)
        action = dist.sample()  # 随机采样一个动作
        return action.item(), dist.log_prob(action), value.squeeze()
    
    def evaluate(self, states, actions):
        """
        评估一批状态-动作对的价值和概率
        
        返回：
        - 动作的对数概率
        - 状态价值
        - 策略的熵（衡量策略的随机性/探索程度）
        """
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()  # 熵鼓励探索
        return log_probs, values.squeeze(), entropy

# 经验收集缓冲区
class RolloutBuffer:
    """
    存储智能体与环境交互产生的经验数据
    包括状态、动作、奖励等信息，用于PPO算法的离线训练
    """
    def __init__(self):
        self.states = []     # 状态
        self.actions = []    # 动作
        self.log_probs = []  # 动作的对数概率
        self.values = []     # 状态价值估计
        self.rewards = []    # 获得的奖励
        self.dones = []      # 回合是否结束
        
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

# PPO智能体（修正版）
class PPOAgent:
    def __init__(self, cfg):
        """初始化PPO智能体"""
        self.cfg = cfg
        # 检测GPU可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if USE_CHINESE:
            print(f"使用设备: {self.device}")
        else:
            print(f"Using device: {self.device}")
        
        # 初始化网络和优化器
        self.policy = PPO(cfg).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        
        # 经验缓冲区
        self.buffer = RolloutBuffer()
        
        # 训练记录
        self.ep_rewards = []  # 每个回合的累积奖励
        self.losses = []      # 每次更新的损失值
        
    def collect_experience(self, env):
        """
        在环境中收集训练数据
        
        与环境交互horizon步，将经验存入缓冲区
        """
        state, _ = env.reset()
        ep_reward = 0
        
        for _ in range(self.cfg.horizon):
            if self.cfg.render:
                env.render()
            
            # 将状态转换为Tensor
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # 选择动作
            action, log_prob, value = self.policy.act(state_tensor)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            
            # 存储经验
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(log_prob)
            self.buffer.values.append(value.item())  # 转换为标量存储
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)
            
            # 更新状态
            state = next_state
            ep_reward += reward
            
            # 处理回合结束
            if done or truncated:
                self.ep_rewards.append(ep_reward)
                ep_reward = 0
                state, _ = env.reset()
                
    def compute_returns_advantages(self, last_value=0):
        """
        计算广义优势估计(GAE)和折扣回报
        
        GAE是一种估计优势函数的方法，平衡了偏差和方差
        优势函数 = 实际回报 - 价值估计，表示动作比平均水平好多少
        
        参数:
            last_value: 最后状态的价值估计
        返回:
            advantages: 每个状态-动作对的优势值
            returns: 每个状态的折扣累积回报
        """
        rewards = np.array(self.buffer.rewards)
        dones = np.array(self.buffer.dones)
        values = np.array(self.buffer.values + [last_value])
        
        # 计算GAE和returns
        # δ_t = r_t + γV(s_{t+1}) - V(s_t)
        deltas = rewards + self.cfg.gamma * values[1:] * (1 - dones) - values[:-1]
        advantages = np.zeros_like(rewards)
        advantage = 0
        
        # 反向计算累积优势
        # A_t = δ_t + γλA_{t+1}
        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage
        
        # 计算returns = advantages + values
        returns = advantages + values[:-1]
        
        # 标准化优势，使其均值为0，方差为1，有助于稳定训练
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return (
            torch.FloatTensor(advantages).to(self.device),
            torch.FloatTensor(returns).to(self.device)
        )
    
    def update_policy(self):
        """
        使用PPO算法更新策略
        
        PPO的核心是限制策略更新的幅度，避免过大的更新导致性能崩溃
        使用重要性采样比率的裁剪来实现这一点
        """
        # 转换数据为Tensor
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device)
        
        # 计算advantages和returns
        advantages, returns = self.compute_returns_advantages()
        
        # 多次进行PPO更新，充分利用收集的数据
        for _ in range(self.cfg.epochs):
            # 随机打乱数据，增加样本多样性
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # 分批次更新，提高内存效率
            for start in range(0, len(states), self.cfg.batch_size):
                end = start + self.cfg.batch_size
                idx = indices[start:end]
                
                # 获取当前批次数据
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # 评估当前策略下的对数概率和价值
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # 计算重要性采样比率 r(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # 计算策略损失 - PPO的核心，使用裁剪目标函数
                # L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.cfg.ppo_eps, 1 + self.cfg.ppo_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失 - 均方误差
                value_loss = 0.5 * (values - batch_returns).pow(2).mean()
                
                # 总损失 = 策略损失 + 价值损失系数 * 价值损失 - 熵系数 * 熵
                # 熵项鼓励探索，防止策略过早收敛
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # 反向传播
                self.optimizer.zero_grad()  # 清除梯度
                loss.backward()             # 计算梯度
                self.optimizer.step()       # 更新参数
                
                self.losses.append(loss.item())
    
    def train(self):
        """
        完整的训练过程
        
        1. 收集经验数据
        2. 更新策略
        3. 重复上述过程
        """
        env = gym.make(self.cfg.env_name)
        for ep in range(self.cfg.max_episodes):
            # 收集经验
            self.collect_experience(env)
            
            # 策略更新
            self.update_policy()
            
            # 清空缓冲区
            self.buffer.clear()
            
            # 输出训练信息
            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(self.ep_rewards[-10:])
                if USE_CHINESE:
                    print(f"回合 {ep+1}/{self.cfg.max_episodes}, 平均奖励: {avg_reward:.2f}")
                else:
                    print(f"Episode {ep+1}/{self.cfg.max_episodes}, Avg Reward: {avg_reward:.2f}")
                
            # 保存模型
            if (ep + 1) % self.cfg.save_freq == 0:
                torch.save(self.policy.state_dict(), f"ppo_cartpole_{ep+1}.pth")
        
        # 保存最终模型
        torch.save(self.policy.state_dict(), "ppo_cartpole_final.pth")
        env.close()
        
    def plot_learning_curve(self):
        """绘制学习曲线，显示训练过程中的奖励和损失变化"""
        plt.figure(figsize=(10, 5))
        
        # 绘制奖励曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.ep_rewards)
        try:
            plt.xlabel('回合数')
            plt.ylabel('回合奖励')
            plt.title('训练奖励曲线')
        except:
            # 如果中文显示失败，使用英文标签
            plt.xlabel('Episodes')
            plt.ylabel('Rewards')
            plt.title('Training Rewards')
        
        # 绘制损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.losses)
        try:
            plt.xlabel('更新步骤')
            plt.ylabel('损失值')
            plt.title('训练损失曲线')
        except:
            # 如果中文显示失败，使用英文标签
            plt.xlabel('Update Steps')
            plt.ylabel('Loss')
            plt.title('Training Losses')
        
        plt.tight_layout()
        plt.savefig('ppo_learning_curve.png')
        plt.show()
        
    def evaluate(self, model_path="ppo_cartpole_final.pth"):
        """
        评估训练好的模型
        
        使用训练好的策略，在环境中运行多个回合并显示效果
        """
        if USE_CHINESE:
            print(f"\n开始评估模型: {model_path}")
        else:
            print(f"\nEvaluating model: {model_path}")
        
        # 加载训练好的模型
        self.policy.load_state_dict(torch.load(model_path))
        self.policy.eval()  # 设置为评估模式
        
        # 创建环境并启用渲染
        env = gym.make(self.cfg.env_name, render_mode="human")
        
        rewards = []
        for ep in range(self.cfg.eval_episodes):
            state, _ = env.reset()
            ep_reward = 0
            done = False
            steps = 0
            
            while not done:
                # 渲染环境
                env.render()
                time.sleep(0.01)  # 减慢渲染速度，便于观察
                
                # 选择动作
                state_tensor = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    logits, _ = self.policy(state_tensor)
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()
                
                # 执行动作
                state, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
                steps += 1
                
                if truncated:
                    break
            
            if USE_CHINESE:
                print(f"评估回合 {ep+1}: 奖励 = {ep_reward}, 步数 = {steps}")
            else:
                print(f"Eval Episode {ep+1}: Reward = {ep_reward}, Steps = {steps}")
            
            rewards.append(ep_reward)
        
        env.close()
        
        # 输出评估结果
        avg_reward = np.mean(rewards)
        if USE_CHINESE:
            print(f"\n评估完成! {self.cfg.eval_episodes}个回合的平均奖励: {avg_reward:.2f}")
        else:
            print(f"\nEvaluation complete! Average reward over {self.cfg.eval_episodes} episodes: {avg_reward:.2f}")
        
        return avg_reward

# 主程序
if __name__ == "__main__":
    # PPO算法工作流程:
    # 1. 智能体与环境交互收集经验
    # 2. 计算优势函数和回报
    # 3. 多次利用同一批数据更新策略，但限制更新幅度
    # 4. 重复以上步骤
    
    cfg = Config()
    agent = PPOAgent(cfg)
    
    if USE_CHINESE:
        print("开始PPO训练...")
        print("算法说明:")
        print("1. PPO(近端策略优化)是一种on-policy策略梯度方法")
        print("2. 它使用裁剪目标函数限制策略更新幅度，提高稳定性")
        print("3. 采用Actor-Critic架构，同时学习策略和价值函数")
        print("4. 使用GAE(广义优势估计)计算更准确的优势函数")
        print("\n开始训练过程...\n")
    else:
        print("Starting PPO training...")
        print("Algorithm overview:")
        print("1. PPO (Proximal Policy Optimization) is an on-policy gradient method")
        print("2. It uses a clipping objective to limit policy updates, improving stability")
        print("3. It adopts an Actor-Critic architecture to learn both policy and value functions")
        print("4. It uses GAE (Generalized Advantage Estimation) for more accurate advantage estimation")
        print("\nStarting training process...\n")
    
    # 训练模型
    agent.train()
    
    # 绘制学习曲线
    agent.plot_learning_curve()
    
    if USE_CHINESE:
        print("训练完成，学习曲线已保存为ppo_learning_curve.png")
        print("\n现在将展示训练好的模型在CartPole环境中的表现...")
    else:
        print("Training complete, learning curve saved as ppo_learning_curve.png")
        print("\nNow showing the trained model's performance in the CartPole environment...")
    
    # 评估训练好的模型
    agent.evaluate()