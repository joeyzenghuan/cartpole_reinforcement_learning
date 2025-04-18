"""
CartPole超参数优化
=================

这个程序通过网格搜索方法寻找CartPole问题最优的超参数组合。
测试不同的超参数组合，包括：
- 学习率(alpha)
- 折扣因子(gamma)
- 初始探索率(epsilon_start)
- 探索率衰减系数(epsilon_decay)
- 状态空间离散化粒度(n_bins)

作者：AI助手
日期：2023
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
from tqdm import tqdm  # 用于显示进度条

# 定义离散化函数
def discretize_state(state, n_bins, bounds):
    """
    将连续状态离散化为箱子索引
    
    参数:
        state: 环境状态
        n_bins: 每个维度的箱子数量
        bounds: 每个维度的上下界 [(低1,高1), (低2,高2), ...]
    
    返回:
        tuple: 离散化的状态索引
    """
    # 为每个状态维度定义值域范围
    indexes = []
    for i, (lower, upper) in enumerate(bounds):
        # 创建线性空间并用数字化函数获取索引
        bins = np.linspace(lower, upper, n_bins)
        index = np.digitize(state[i], bins)
        # 确保索引不会超出范围
        index = min(index, n_bins - 1)
        indexes.append(index)
    
    return tuple(indexes)

# 定义动作选择函数
def select_action(state, q_table, epsilon):
    """
    使用epsilon-greedy策略选择动作
    """
    if np.random.random() < epsilon:
        return np.random.randint(0, 2)
    else:
        return np.argmax(q_table[state])

# 定义单次训练函数
def train_once(params, max_episodes=200, max_steps=200, verbose=False):
    """
    使用给定参数训练一次CartPole，并返回性能指标
    
    参数:
        params: 字典，包含各超参数
        max_episodes: 最大训练回合数
        max_steps: 每回合最大步数
        verbose: 是否打印详细信息
    
    返回:
        dict: 包含训练结果的字典
    """
    # 解包参数
    n_bins = params['n_bins']
    alpha = params['alpha'] 
    gamma = params['gamma']
    epsilon_start = params['epsilon_start']
    epsilon_decay = params['epsilon_decay']
    epsilon_end = params['epsilon_end']
    
    # 设置环境和状态空间范围
    env = gym.make('CartPole-v1', render_mode=None)  # 不显示训练过程
    n_dims = 4
    state_bounds = [
        (-2.4, 2.4),    # 小车位置范围
        (-4.0, 4.0),    # 小车速度范围
        (-0.2095, 0.2095),  # 杆子角度范围
        (-4.0, 4.0)     # 杆子角速度范围
    ]
    
    # 创建Q表
    q_table = np.zeros([n_bins] * n_dims + [env.action_space.n])
    
    # 训练过程
    episode_rewards = []
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        state = discretize_state(state, n_bins, state_bounds)
        total_reward = 0
        
        # 更新探索率
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        for step in range(max_steps):
            # 选择动作
            action = select_action(state, q_table, epsilon)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, n_bins, state_bounds)
            total_reward += reward
            
            # 是否结束
            done = terminated or truncated
            
            # Q-learning更新
            old_value = q_table[state + (action,)]
            next_best = np.max(q_table[next_state])
            td_target = reward + gamma * next_best
            td_error = td_target - old_value
            new_value = old_value + alpha * td_error
            q_table[state + (action,)] = new_value
            
            # 更新状态
            state = next_state
            
            if done:
                break
                
        episode_rewards.append(total_reward)
        if verbose and episode % 20 == 0:
            print(f"回合 {episode}/{max_episodes}, 总奖励: {total_reward:.2f}, 探索率: {epsilon:.4f}")
    
    env.close()
    
    # 计算性能指标
    avg_reward = np.mean(episode_rewards)
    last_100_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else avg_reward
    max_reward = np.max(episode_rewards)
    
    # 计算收敛速度 (找到首次达到195分的回合)
    convergence_episode = next((i for i, r in enumerate(episode_rewards) if r >= 195), max_episodes)
    
    # 计算稳定性 (最后100回合中奖励>=195的比例)
    stability = np.mean([1 if r >= 195 else 0 for r in episode_rewards[-100:]]) if len(episode_rewards) >= 100 else 0
    
    return {
        'params': params,
        'avg_reward': avg_reward,
        'last_100_avg': last_100_avg,
        'max_reward': max_reward,
        'convergence_episode': convergence_episode,
        'stability': stability,
        'episode_rewards': episode_rewards
    }

# 定义网格搜索函数
def grid_search():
    # 设置要测试的参数网格
    param_grid = {
        'n_bins': [8, 10, 12],  # 状态离散化粒度
        'alpha': [0.05, 0.1, 0.2],  # 学习率
        'gamma': [0.95, 0.99],  # 折扣因子
        'epsilon_start': [0.3, 0.5, 0.7],  # 初始探索率
        'epsilon_decay': [0.98, 0.99, 0.995],  # 探索率衰减
        'epsilon_end': [0.01]  # 最小探索率（固定）
    }
    
    # 生成所有参数组合
    keys = param_grid.keys()
    param_combinations = [dict(zip(keys, values)) for values in product(*param_grid.values())]
    
    print(f"将测试 {len(param_combinations)} 种参数组合")
    
    # 存储结果
    results = []
    
    # 对每种参数组合进行训练和评估
    for i, params in enumerate(tqdm(param_combinations)):
        print(f"\n测试参数组合 {i+1}/{len(param_combinations)}:")
        print(params)
        
        # 为每组参数训练多次以减少随机性影响
        n_trials = 3
        trial_results = []
        
        for trial in range(n_trials):
            result = train_once(params)
            trial_results.append(result)
        
        # 平均多次训练的结果
        avg_result = {
            'params': params,
            'avg_reward': np.mean([r['avg_reward'] for r in trial_results]),
            'last_100_avg': np.mean([r['last_100_avg'] for r in trial_results]),
            'max_reward': np.mean([r['max_reward'] for r in trial_results]),
            'convergence_episode': np.mean([r['convergence_episode'] for r in trial_results]),
            'stability': np.mean([r['stability'] for r in trial_results])
        }
        
        results.append(avg_result)
        
        print(f"平均结果: 平均奖励={avg_result['avg_reward']:.2f}, " 
              f"收敛回合={avg_result['convergence_episode']:.1f}, "
              f"稳定性={avg_result['stability']:.2f}")
    
    return results

# 绘制结果函数
def plot_results(results):
    # 转换为DataFrame以便于分析
    df = pd.DataFrame(results)
    
    # 按平均奖励排序
    df_sorted = df.sort_values('last_100_avg', ascending=False)
    
    # 显示最佳参数
    best_result = df_sorted.iloc[0]
    print("\n最佳参数组合:")
    for key, value in best_result['params'].items():
        print(f"{key}: {value}")
    
    print(f"\n性能指标:")
    print(f"最后100回合平均奖励: {best_result['last_100_avg']:.2f}")
    print(f"收敛回合: {best_result['convergence_episode']:.1f}")
    print(f"稳定性: {best_result['stability']:.2f}")
    
    # 绘制参数影响图
    plt.figure(figsize=(15, 10))
    
    # 1. 学习率影响
    plt.subplot(2, 3, 1)
    alpha_groups = df.groupby('params').apply(lambda x: (x['params']['alpha'], x['last_100_avg'])).reset_index()
    alpha_values = sorted(list(set([x[0] for x in alpha_groups[0]])))
    alpha_rewards = [np.mean([x[1] for x in alpha_groups[0] if x[0] == a]) for a in alpha_values]
    plt.plot(alpha_values, alpha_rewards, 'o-')
    plt.xlabel('学习率 (alpha)')
    plt.ylabel('平均奖励')
    plt.title('学习率影响')
    
    # 2. 折扣因子影响
    plt.subplot(2, 3, 2)
    gamma_groups = df.groupby('params').apply(lambda x: (x['params']['gamma'], x['last_100_avg'])).reset_index()
    gamma_values = sorted(list(set([x[0] for x in gamma_groups[0]])))
    gamma_rewards = [np.mean([x[1] for x in gamma_groups[0] if x[0] == g]) for g in gamma_values]
    plt.plot(gamma_values, gamma_rewards, 'o-')
    plt.xlabel('折扣因子 (gamma)')
    plt.ylabel('平均奖励')
    plt.title('折扣因子影响')
    
    # 3. 初始探索率影响
    plt.subplot(2, 3, 3)
    eps_start_groups = df.groupby('params').apply(lambda x: (x['params']['epsilon_start'], x['last_100_avg'])).reset_index()
    eps_start_values = sorted(list(set([x[0] for x in eps_start_groups[0]])))
    eps_start_rewards = [np.mean([x[1] for x in eps_start_groups[0] if x[0] == e]) for e in eps_start_values]
    plt.plot(eps_start_values, eps_start_rewards, 'o-')
    plt.xlabel('初始探索率 (epsilon_start)')
    plt.ylabel('平均奖励')
    plt.title('初始探索率影响')
    
    # 4. 探索率衰减影响
    plt.subplot(2, 3, 4)
    eps_decay_groups = df.groupby('params').apply(lambda x: (x['params']['epsilon_decay'], x['last_100_avg'])).reset_index()
    eps_decay_values = sorted(list(set([x[0] for x in eps_decay_groups[0]])))
    eps_decay_rewards = [np.mean([x[1] for x in eps_decay_groups[0] if x[0] == e]) for e in eps_decay_values]
    plt.plot(eps_decay_values, eps_decay_rewards, 'o-')
    plt.xlabel('探索率衰减 (epsilon_decay)')
    plt.ylabel('平均奖励')
    plt.title('探索率衰减影响')
    
    # 5. 状态离散化粒度影响
    plt.subplot(2, 3, 5)
    bins_groups = df.groupby('params').apply(lambda x: (x['params']['n_bins'], x['last_100_avg'])).reset_index()
    bins_values = sorted(list(set([x[0] for x in bins_groups[0]])))
    bins_rewards = [np.mean([x[1] for x in bins_groups[0] if x[0] == b]) for b in bins_values]
    plt.plot(bins_values, bins_rewards, 'o-')
    plt.xlabel('状态离散化粒度 (n_bins)')
    plt.ylabel('平均奖励')
    plt.title('状态离散化粒度影响')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_analysis.png')
    
    return best_result

# 使用最佳参数训练并可视化结果
def train_with_best_params(best_params, episodes=500):
    print("\n使用最佳参数进行训练...")
    
    # 创建可视化环境
    env = gym.make('CartPole-v1', render_mode='human')
    
    # 设置参数
    params = best_params.copy()
    
    # 训练并评估
    result = train_once(params, max_episodes=episodes, verbose=True)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(result['episode_rewards'], alpha=0.6, label='原始奖励')
    
    # 添加滑动平均
    window_size = 10
    rewards_smoothed = np.convolve(result['episode_rewards'], np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(result['episode_rewards'])), rewards_smoothed, linewidth=2, 
             label=f'{window_size}回合滑动平均')
    
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title(f'最佳参数组合的学习曲线 (n_bins={params["n_bins"]}, alpha={params["alpha"]}, gamma={params["gamma"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('best_params_learning_curve.png')
    
    # 展示最后10回合的性能
    print(f"最后10回合平均奖励: {np.mean(result['episode_rewards'][-10:]):.2f}")
    
    return result

# 主函数
def main():
    print("开始CartPole超参数优化...")
    
    # 网格搜索
    results = grid_search()
    
    # 分析结果
    best_result = plot_results(results)
    
    # 用最佳参数训练并展示
    train_with_best_params(best_result['params'])
    
    print("\n超参数优化完成！")
    print("最佳参数组合已保存，学习曲线已绘制。")

if __name__ == "__main__":
    main() 