"""
CartPole最优参数训练
==================

使用经过优化的超参数训练CartPole问题，并展示训练效果。
这个脚本使用我们通过网格搜索找到的最优超参数组合，训练一个高性能的CartPole控制智能体。

作者：AI助手
日期：2023
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# 超参数（经过优化的最佳组合）
OPTIMAL_PARAMS = {
    'n_bins': 12,          # 状态空间离散化粒度 - 更精细的网格
    'alpha': 0.1,          # 学习率 - 平衡学习速度和稳定性
    'gamma': 0.99,         # 折扣因子 - 注重长期奖励
    'epsilon_start': 0.3,  # 初始探索率 - 适度的初始探索
    'epsilon_decay': 0.995,# 探索率衰减 - 缓慢减少探索以获得更好的学习
    'epsilon_end': 0.01    # 最终探索率 - 保持少量探索
}

# 定义状态离散化函数
def discretize_state(state, n_bins):
    """
    将连续的状态值转换为离散的箱子索引
    
    参数:
        state: CartPole的观测状态 [位置, 速度, 角度, 角速度]
        n_bins: 每个维度的离散化格数
        
    返回:
        tuple: 离散化后的状态索引
    """
    # 定义每个维度的值域范围
    state_bounds = [
        (-2.4, 2.4),       # 小车位置范围
        (-4.0, 4.0),       # 小车速度范围
        (-0.2095, 0.2095), # 杆子角度范围
        (-4.0, 4.0)        # 杆子角速度范围
    ]
    
    # 离散化每个维度
    indices = []
    for i, (lower, upper) in enumerate(state_bounds):
        bins = np.linspace(lower, upper, n_bins)
        index = np.digitize(state[i], bins)
        index = min(index, n_bins - 1)  # 确保索引不超出边界
        indices.append(index)
    
    return tuple(indices)

# 定义动作选择函数
def select_action(state, q_table, epsilon):
    """
    使用epsilon-greedy策略选择动作
    
    参数:
        state: 当前状态
        q_table: Q值表
        epsilon: 探索率
        
    返回:
        int: 选择的动作 (0表示向左, 1表示向右)
    """
    # 探索：以epsilon的概率随机选择动作
    if np.random.random() < epsilon:
        return np.random.randint(0, 2)
    # 利用：选择当前状态下Q值最高的动作
    else:
        return np.argmax(q_table[state])

# 训练函数
def train_agent(params=OPTIMAL_PARAMS, max_episodes=500, render=False):
    """
    训练CartPole智能体
    
    参数:
        params: 超参数字典
        max_episodes: 训练回合数
        render: 是否显示训练过程
        
    返回:
        tuple: (Q表, 奖励历史)
    """
    # 创建环境
    render_mode = 'human' if render else None
    env = gym.make('CartPole-v1', render_mode=render_mode)
    
    # 从参数中获取超参数
    n_bins = params['n_bins']
    alpha = params['alpha']
    gamma = params['gamma']
    epsilon_start = params['epsilon_start']
    epsilon_decay = params['epsilon_decay']
    epsilon_end = params['epsilon_end']
    
    # 初始化Q表
    n_dims = 4  # 状态空间维度数
    q_table = np.zeros([n_bins] * n_dims + [env.action_space.n])
    
    # 用于记录训练过程
    episode_rewards = []
    
    print("开始训练...")
    print(f"使用参数: n_bins={n_bins}, alpha={alpha}, gamma={gamma}, " 
          f"epsilon_start={epsilon_start}, epsilon_decay={epsilon_decay}")
    
    # 训练过程
    for episode in range(max_episodes):
        # 重置环境
        state, _ = env.reset()
        state = discretize_state(state, n_bins)
        total_reward = 0
        done = False
        
        # 计算当前回合的探索率
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        # 单回合训练
        while not done:
            # 选择动作
            action = select_action(state, q_table, epsilon)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, n_bins)
            done = terminated or truncated
            total_reward += reward
            
            # Q-learning更新
            old_value = q_table[state + (action,)]
            next_best = np.max(q_table[next_state])
            td_target = reward + gamma * next_best
            td_error = td_target - old_value
            new_value = old_value + alpha * td_error
            q_table[state + (action,)] = new_value
            
            # 移动到下一个状态
            state = next_state
            
            # 控制渲染速度
            if render:
                sleep(0.01)
        
        # 记录奖励
        episode_rewards.append(total_reward)
        
        # 显示进度
        if episode % 20 == 0 or episode == max_episodes - 1:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"回合 {episode+1}/{max_episodes}, 奖励: {total_reward}, "
                  f"最近平均: {avg_reward:.1f}, 探索率: {epsilon:.3f}")
    
    env.close()
    
    print(f"训练完成! 最后100回合平均奖励: {np.mean(episode_rewards[-100:]):.1f}")
    return q_table, episode_rewards

# 评估函数
def evaluate_agent(q_table, n_bins, n_episodes=10, render=True, delay=0.01):
    """
    评估已训练的智能体
    
    参数:
        q_table: 训练好的Q表
        n_bins: 状态空间离散化粒度
        n_episodes: 评估回合数
        render: 是否显示评估过程
        delay: 每步之间的延迟（秒）
    """
    # 创建环境
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    rewards = []
    print("\n===== 评估智能体 =====")
    
    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode+1000)  # 使用不同种子以测试泛化能力
        state = discretize_state(state, n_bins)
        total_reward = 0
        done = False
        
        # 运行一个回合
        while not done:
            # 使用贪婪策略（不再探索）
            action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, n_bins)
            done = terminated or truncated
            total_reward += reward
            
            state = next_state
            
            if render:
                sleep(delay)  # 控制显示速度
        
        rewards.append(total_reward)
        print(f"测试回合 {episode+1}/{n_episodes}, 总奖励: {total_reward}")
    
    env.close()
    
    # 显示评估结果
    print(f"\n评估结果: 平均奖励 = {np.mean(rewards):.1f}")
    return rewards

# 绘制学习曲线
def plot_learning_curve(rewards):
    """绘制训练过程中的奖励变化"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.6, label='回合奖励')
    
    # 添加滑动平均线
    window_size = 20
    if len(rewards) >= window_size:
        rewards_smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), rewards_smoothed, 
                 linewidth=2, label=f'{window_size}回合滑动平均')
    
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.title('CartPole训练学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cartpole_learning_curve.png')
    plt.show()

# 保存Q表
def save_q_table(q_table, filename='cartpole_q_table.npy'):
    """保存训练好的Q表"""
    np.save(filename, q_table)
    print(f"Q表已保存到 {filename}")

# 加载Q表
def load_q_table(filename='cartpole_q_table.npy'):
    """加载保存的Q表"""
    try:
        q_table = np.load(filename)
        print(f"Q表已从 {filename} 加载")
        return q_table
    except:
        print(f"无法加载Q表文件 {filename}")
        return None

# 主函数
def main():
    """主函数"""
    print("===== CartPole最优参数训练 =====")
    
    # 训练智能体
    print("\n1. 训练阶段")
    q_table, rewards = train_agent(OPTIMAL_PARAMS, max_episodes=500, render=False)
    
    # 保存Q表
    save_q_table(q_table)
    
    # 绘制学习曲线
    plot_learning_curve(rewards)
    
    # 评估智能体
    print("\n2. 评估阶段")
    evaluate_agent(q_table, OPTIMAL_PARAMS['n_bins'], n_episodes=5, render=True)
    
    print("\n训练和评估完成!")
    print("学习曲线已保存为 'cartpole_learning_curve.png'")

if __name__ == "__main__":
    main() 


# 最佳参数组合:
# n_bins: 12
# alpha: 0.2
# gamma: 0.95
# epsilon_start: 0.3
# epsilon_decay: 0.995
# epsilon_end: 0.01

# 性能指标:
# 最后100回合平均奖励: 51.59
# 收敛回合: 200.0
# 稳定性: 0.00    