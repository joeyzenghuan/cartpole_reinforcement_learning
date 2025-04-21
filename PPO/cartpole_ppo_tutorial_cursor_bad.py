"""
强化学习进阶教程：使用PPO（Proximal Policy Optimization）算法训练CartPole平衡问题
=========================================================================

这是一个强化学习进阶教程，使用现代的PPO算法来解决经典的CartPole问题。
PPO是一种基于策略梯度的强化学习算法，由OpenAI在2017年提出，
它在稳定性和采样效率方面表现出色，被广泛应用于各种强化学习任务。

本教程将详细展示PPO算法的实现过程，并与Q-learning等传统方法进行对比。

作者：AI助手
日期：2023
"""

# 导入必要的库
import gymnasium as gym  # 强化学习环境库
import numpy as np       # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
import torch             # PyTorch深度学习库
import torch.nn as nn    # 神经网络模块
import torch.optim as optim  # 优化器
from torch.distributions import Categorical  # 用于采样动作
import time              # 用于测量执行时间
import os              # 用于文件操作

# 设置随机种子以确保结果可重现
np.random.seed(42)
torch.manual_seed(42)

# 第一步：创建CartPole环境
# ======================
env = gym.make('CartPole-v1')
env.reset(seed=42)  # 设置环境的随机种子

# 第二步：定义PPO的神经网络模型
# ==========================
class PPONetwork(nn.Module):
    """
    PPO神经网络模型，包含策略网络和价值网络
    策略网络输出动作的概率分布，价值网络估计状态的价值
    """
    def __init__(self, state_dim, action_dim):
        """
        初始化神经网络
        
        参数:
            state_dim (int): 状态空间的维度
            action_dim (int): 动作空间的维度
        """
        super(PPONetwork, self).__init__()
        
        # 策略网络（Actor）- 使用更深的网络结构
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络（Critic）- 独立的网络结构
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        """
        前向传播函数
        
        参数:
            state (torch.Tensor): 环境状态
            
        返回:
            action_probs (torch.Tensor): 动作概率分布
            state_value (torch.Tensor): 状态价值估计
        """
        action_probs = self.actor(state)
        state_value = self.critic(state)
        
        return action_probs, state_value
    
    def get_action(self, state, action=None):
        """
        根据状态选择动作
        
        参数:
            state (numpy.array): 环境状态
            action (int, optional): 如果提供，计算此动作的对数概率
            
        返回:
            action (int): 选择的动作
            action_log_prob (torch.Tensor): 动作的对数概率
            state_value (torch.Tensor): 状态的价值估计
            entropy (torch.Tensor): 策略的熵，用于鼓励探索
        """
        # 将NumPy数组转换为PyTorch张量
        state = torch.FloatTensor(state)
        
        # 获取动作概率和状态价值
        action_probs, state_value = self.forward(state)
        
        # 创建动作的概率分布
        dist = Categorical(action_probs)
        
        # 如果没有提供动作，就从分布中采样一个动作
        selected_action = action if action is not None else dist.sample().item()
        
        # 计算动作的对数概率
        action_log_prob = dist.log_prob(torch.tensor(selected_action))
        
        # 计算策略的熵（用于鼓励探索）
        entropy = dist.entropy()
        
        return selected_action, action_log_prob, state_value, entropy

# 第三步：定义PPO算法的主要类
# =========================
class PPO:
    """
    实现PPO（Proximal Policy Optimization）算法
    """
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, 
                 gae_lambda=0.95, epsilon=0.2, value_coef=0.5, entropy_coef=0.01, 
                 num_epochs=10, batch_size=128, clip_grad_norm=0.5):
        """
        初始化PPO算法
        
        参数:
            state_dim (int): 状态空间的维度
            action_dim (int): 动作空间的维度
            lr_actor (float): Actor网络的学习率
            lr_critic (float): Critic网络的学习率
            gamma (float): 折扣因子
            gae_lambda (float): GAE参数
            epsilon (float): PPO裁剪参数
            value_coef (float): 价值损失的权重系数
            entropy_coef (float): 熵正则化的权重系数
            num_epochs (int): 每批数据的更新次数
            batch_size (int): 小批量大小
            clip_grad_norm (float): 梯度裁剪阈值
        """
        self.gamma = gamma              # 折扣因子
        self.gae_lambda = gae_lambda    # GAE参数
        self.epsilon = epsilon          # PPO裁剪参数
        self.value_coef = value_coef    # 价值损失权重
        self.entropy_coef = entropy_coef  # 熵正则化权重
        self.num_epochs = num_epochs    # 每批数据的更新次数
        self.batch_size = batch_size    # 小批量大小
        self.clip_grad_norm = clip_grad_norm  # 梯度裁剪阈值
        
        # 创建策略网络
        self.network = PPONetwork(state_dim, action_dim)
        
        # 分别为actor和critic设置优化器
        self.optimizer = optim.Adam([
            {'params': self.network.actor.parameters(), 'lr': lr_actor},
            {'params': self.network.critic.parameters(), 'lr': lr_critic}
        ])
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        更新PPO网络参数
        
        参数:
            states (numpy.array): 收集的状态
            actions (numpy.array): 执行的动作
            old_log_probs (numpy.array): 旧策略下动作的对数概率
            returns (numpy.array): 计算的回报
            advantages (numpy.array): 优势估计
        
        返回:
            policy_loss (float): 策略损失
            value_loss (float): 价值损失
            entropy (float): 策略的熵
        """
        # 将NumPy数组转换为PyTorch张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 创建数据集以进行小批量更新
        dataset_size = states.shape[0]
        
        # 记录损失
        policy_losses = []
        value_losses = []
        entropies = []
        
        # 对数据进行多个epoch的更新
        for _ in range(self.num_epochs):
            # 生成随机索引
            indices = torch.randperm(dataset_size)
            
            # 按批次处理数据
            for start_idx in range(0, dataset_size, self.batch_size):
                # 获取批次索引
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                # 提取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 获取当前策略的动作概率和状态价值
                action_probs, state_values = self.network(batch_states)
                dist = Categorical(action_probs)
                
                # 计算当前策略下动作的对数概率
                new_log_probs = dist.log_prob(batch_actions)
                
                # 计算策略的熵
                entropy = dist.entropy().mean()
                
                # 计算比率 r(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算裁剪的目标函数
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                
                # 策略损失: 取两者中的较小值，确保策略更新不会过大
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失: 使用MSE损失函数计算估计价值与实际回报的差距
                value_loss = nn.MSELoss()(state_values.squeeze(-1), batch_returns)
                
                # 总损失 = 策略损失 + 价值系数 * 价值损失 - 熵系数 * 熵
                # 减去熵是为了鼓励探索（熵越大，探索越多）
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 执行反向传播和优化
                self.optimizer.zero_grad()  # 清除之前的梯度
                total_loss.backward()       # 计算梯度
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
                
                self.optimizer.step()       # 更新参数
                
                # 记录损失
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)
    
    def compute_returns_and_advantages(self, rewards, values, dones):
        """
        计算每个时间步的回报和优势
        
        参数:
            rewards (list): 每个时间步获得的奖励
            values (list): 价值网络对每个状态的估计价值
            dones (list): 表示每个时间步是否结束回合
            
        返回:
            returns (numpy.array): 每个时间步的回报
            advantages (numpy.array): 每个时间步的优势
        """
        returns = []
        advantages = []
        gae = 0  # 初始化广义优势估计(GAE)
        
        # 从最后一个时间步开始，反向计算回报和优势
        for t in reversed(range(len(rewards))):
            # 如果是最后一个时间步或回合结束，下一状态的价值为0
            if t == len(rewards) - 1 or dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # 计算TD误差 δ = r + γV(s') - V(s)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # 计算GAE: A(s,a) = δ + γλA(s',a')
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            # 将优势添加到列表的开头（因为我们是从后向前计算的）
            advantages.insert(0, gae)
            
            # 计算回报：R = A + V
            returns.insert(0, gae + values[t])
        
        return np.array(returns), np.array(advantages)

# 第四步：设置训练参数
# =================
state_dim = env.observation_space.shape[0]  # CartPole状态空间维度为4
action_dim = env.action_space.n             # CartPole动作空间维度为2
max_episodes = 500                          # 最大训练回合数
max_steps = 500                             # 每个回合的最大步数
update_interval = 20                        # 数据收集间隔
target_reward = 475                         # 目标奖励（接近满分500）

# 创建PPO代理
agent = PPO(state_dim, action_dim, 
           lr_actor=0.0005,                # Actor学习率
           lr_critic=0.001,                # Critic学习率
           gamma=0.99,                     # 折扣因子
           gae_lambda=0.95,                # GAE参数
           epsilon=0.2,                    # PPO裁剪参数
           value_coef=0.5,                 # 价值损失权重
           entropy_coef=0.01,              # 熵正则化权重
           num_epochs=4,                   # 每批数据的更新次数
           batch_size=64,                  # 小批量大小
           clip_grad_norm=0.5              # 梯度裁剪阈值
)

# 用于记录训练过程
episode_rewards = []
average_rewards = []
best_reward = 0

# 引入额外的探索参数 - 初始温度
exploration_temp = 1.0
min_exploration_temp = 0.1
temp_decay = 0.995  # 温度衰减率

# 第五步：开始训练过程
# =================
print("开始PPO强化学习训练...")
print(f"状态空间维度: {state_dim}")
print(f"动作空间维度: {action_dim}")
print(f"最大训练回合数: {max_episodes}")
print(f"每回合最大步数: {max_steps}")
print(f"参数更新间隔: {update_interval}回合")
print("============================")

start_time = time.time()  # 记录开始时间

for episode in range(max_episodes):
    # 用于收集一个更新周期的数据
    if episode % update_interval == 0:
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
    
    # 重置环境
    state, _ = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        # 选择动作
        with torch.no_grad():
            action, log_prob, value, _ = agent.network.get_action(state)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 改善奖励函数 - 引入位置和角度的奖励
        # CartPole的状态: [位置, 速度, 杆角度, 角速度]
        pos, vel, angle, ang_vel = next_state
        
        # 根据位置和角度给予额外奖励，保持杆子直立且小车在中心
        pos_reward = 0.1 * (1.0 - min(abs(pos), 2.4) / 2.4)
        angle_reward = 0.1 * (1.0 - min(abs(angle), 0.2) / 0.2)
        custom_reward = reward + pos_reward + angle_reward
        
        # 收集数据
        states.append(state)
        actions.append(action)
        rewards.append(custom_reward)
        log_probs.append(log_prob.detach().numpy())
        values.append(value.detach().numpy())
        dones.append(done)
        
        # 更新状态和累积奖励
        state = next_state
        episode_reward += reward  # 使用原始奖励来记录性能
        
        # 如果回合结束，跳出循环
        if done:
            break
    
    # 记录回合奖励
    episode_rewards.append(episode_reward)
    
    # 温度衰减，减少探索
    exploration_temp = max(min_exploration_temp, exploration_temp * temp_decay)
    
    # 计算最近10个回合的平均奖励
    if len(episode_rewards) >= 10:
        avg_reward = np.mean(episode_rewards[-10:])
        average_rewards.append(avg_reward)
        
        # 打印训练进度
        elapsed_time = time.time() - start_time
        print(f"回合 {episode+1}/{max_episodes}, 奖励: {episode_reward}, 10回合平均: {avg_reward:.2f}, 探索温度: {exploration_temp:.2f}, 用时: {elapsed_time:.2f}秒")
        
        # 如果达到目标奖励，提前结束训练
        if avg_reward > target_reward:
            print(f"环境已解决! 10回合平均奖励: {avg_reward:.2f}")
            break
    else:
        average_rewards.append(episode_reward)
        print(f"回合 {episode+1}/{max_episodes}, 奖励: {episode_reward}")
    
    # 如果收集了足够的数据，更新策略
    if (episode + 1) % update_interval == 0 or episode == max_episodes - 1:
        # 将列表转换为NumPy数组
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        log_probs = np.array(log_probs)
        values = np.array(values)
        dones = np.array(dones)
        
        # 计算回报和优势
        returns, advantages = agent.compute_returns_and_advantages(rewards, values, dones)
        
        # 标准化优势（使均值为0，方差为1）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新策略
        policy_loss, value_loss, entropy = agent.update(states, actions, log_probs, returns, advantages)
        
        print(f"策略更新 - 策略损失: {policy_loss:.4f}, 价值损失: {value_loss:.4f}, 熵: {entropy:.4f}")
        
        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(agent.network.state_dict(), 'best_ppo_model.pth')
            print(f"保存新的最佳模型，平均奖励: {best_reward:.2f}")

# 加载最佳模型
if os.path.exists('best_ppo_model.pth'):
    agent.network.load_state_dict(torch.load('best_ppo_model.pth'))
    print("已加载最佳模型")

# 第六步：可视化训练结果
# ===================
plt.figure(figsize=(12, 6))

# 绘制每回合奖励
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.axhline(y=target_reward, color='r', linestyle='--', label=f'目标奖励: {target_reward}')
plt.xlabel('回合')
plt.ylabel('奖励')
plt.title('每回合奖励')
plt.legend()

# 绘制平均奖励曲线
plt.subplot(1, 2, 2)
plt.plot(average_rewards)
plt.axhline(y=target_reward, color='r', linestyle='--', label=f'目标奖励: {target_reward}')
plt.xlabel('回合')
plt.ylabel('10回合平均奖励')
plt.title('训练曲线')
plt.legend()

plt.tight_layout()
plt.savefig('ppo_learning_curve.png')
plt.show()

# 第七步：测试训练好的模型
# =====================
# 重新创建环境进行测试
test_env = gym.make('CartPole-v1', render_mode='human')
test_episodes = 5
test_rewards = []

print("\n===== 测试训练好的模型 =====")
for episode in range(test_episodes):
    state, _ = test_env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # 选择动作（测试时不需要计算log_prob和value）
        action, _, _, _ = agent.network.get_action(state)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        
        # 更新状态和累积奖励
        state = next_state
        episode_reward += reward
    
    test_rewards.append(episode_reward)
    print(f"测试回合 {episode+1}/{test_episodes}, 奖励: {episode_reward}")

print(f"测试平均奖励: {np.mean(test_rewards):.2f}")

# 关闭环境
env.close()
test_env.close()

# 第八步：分析PPO与Q-learning的区别
# ==============================
print("\n===== PPO与Q-learning的对比分析 =====")
print("1. 模型表示:")
print("   - Q-learning: 使用Q表存储每个状态-动作对的价值")
print("   - PPO: 使用神经网络直接表示策略和价值函数，可以处理连续状态空间")
print("\n2. 更新机制:")
print("   - Q-learning: 基于时序差分(TD)学习，更新单个状态-动作对的价值")
print("   - PPO: 基于策略梯度，通过裁剪目标函数防止过大的策略更新")
print("\n3. 探索策略:")
print("   - Q-learning: 通常使用ε-贪心策略（固定概率随机探索）")
print("   - PPO: 通过最大化策略熵来鼓励探索，随着训练自然减少探索")
print("\n4. 采样效率:")
print("   - Q-learning: 样本利用率低，每个样本只用于更新一个Q值")
print("   - PPO: 样本利用率高，同一批数据可以用于多次更新")
print("\n5. 收敛性:")
print("   - Q-learning: 在简单问题上收敛稳定，但在复杂问题上可能不稳定")
print("   - PPO: 设计上更稳定，通过裁剪概率比来防止灾难性的策略崩溃")
print("\n6. 应用范围:")
print("   - Q-learning: 适合离散动作空间的小型问题")
print("   - PPO: 适用于离散和连续动作空间，以及大规模问题")

print("\n===== 进阶学习方向 =====")
print("1. 探索其他现代强化学习算法: SAC, TD3, TRPO等")
print("2. 应用到更复杂的环境: MuJoCo, Atari游戏等")
print("3. 研究多智能体强化学习和元强化学习")
print("4. 探索基于模型的强化学习方法")
print("5. 将强化学习应用到实际问题：机器人控制、资源调度等") 