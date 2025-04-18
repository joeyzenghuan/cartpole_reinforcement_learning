"""
CartPole最优超参数查找 - 简化版
============================

这个程序通过有限的参数组合测试，快速找出CartPole问题较优的超参数组合。
测试参数包括：
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
import time

# 定义离散化函数
def discretize_state(state, n_bins, bounds):
    """将连续状态离散化为箱子索引"""
    indexes = []
    for i, (lower, upper) in enumerate(bounds):
        bins = np.linspace(lower, upper, n_bins)
        index = np.digitize(state[i], bins)
        index = min(index, n_bins - 1)
        indexes.append(index)
    
    return tuple(indexes)

# 定义动作选择函数
def select_action(state, q_table, epsilon):
    """使用epsilon-greedy策略选择动作"""
    if np.random.random() < epsilon:
        return np.random.randint(0, 2)
    else:
        return np.argmax(q_table[state])

# 定义单次训练函数
def train_once(params, max_episodes=200, max_steps=200, verbose=False):
    """使用给定参数训练一次CartPole，并返回性能指标"""
    # 解包参数
    n_bins = params['n_bins']
    alpha = params['alpha'] 
    gamma = params['gamma']
    epsilon_start = params['epsilon_start']
    epsilon_decay = params['epsilon_decay']
    epsilon_end = params['epsilon_end']
    
    # 设置环境和状态空间范围
    env = gym.make('CartPole-v1', render_mode=None)
    n_dims = 4
    state_bounds = [
        (-2.4, 2.4),       # 小车位置范围
        (-4.0, 4.0),       # 小车速度范围
        (-0.2095, 0.2095), # 杆子角度范围
        (-4.0, 4.0)        # 杆子角速度范围
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
            print(f"回合 {episode+1}/{max_episodes}, 总奖励: {total_reward:.2f}, 探索率: {epsilon:.4f}")
    
    env.close()
    
    # 计算性能指标
    avg_reward = np.mean(episode_rewards)
    last_100_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else avg_reward
    
    # 计算收敛速度 (找到首次达到195分的回合)
    convergence_episode = next((i for i, r in enumerate(episode_rewards) if r >= 195), max_episodes)
    
    # 计算稳定性 (最后100回合中奖励>=195的比例)
    stability = np.mean([1 if r >= 195 else 0 for r in episode_rewards[-100:]]) if len(episode_rewards) >= 100 else 0
    
    return {
        'params': params,
        'avg_reward': avg_reward,
        'last_100_avg': last_100_avg,
        'convergence_episode': convergence_episode,
        'stability': stability,
        'episode_rewards': episode_rewards
    }

# 简化的超参数测试
def simple_optimization():
    # 设置要测试的参数组合（仅测试少量关键组合）
    param_sets = [
        # 默认参数组合
        {'n_bins': 10, 'alpha': 0.1, 'gamma': 0.99, 'epsilon_start': 0.5, 'epsilon_decay': 0.99, 'epsilon_end': 0.01},
        
        # 测试不同的学习率
        {'n_bins': 10, 'alpha': 0.05, 'gamma': 0.99, 'epsilon_start': 0.5, 'epsilon_decay': 0.99, 'epsilon_end': 0.01},
        {'n_bins': 10, 'alpha': 0.2, 'gamma': 0.99, 'epsilon_start': 0.5, 'epsilon_decay': 0.99, 'epsilon_end': 0.01},
        
        # 测试不同的折扣因子
        {'n_bins': 10, 'alpha': 0.1, 'gamma': 0.95, 'epsilon_start': 0.5, 'epsilon_decay': 0.99, 'epsilon_end': 0.01},
        
        # 测试不同的探索率
        {'n_bins': 10, 'alpha': 0.1, 'gamma': 0.99, 'epsilon_start': 0.3, 'epsilon_decay': 0.99, 'epsilon_end': 0.01},
        {'n_bins': 10, 'alpha': 0.1, 'gamma': 0.99, 'epsilon_start': 0.7, 'epsilon_decay': 0.99, 'epsilon_end': 0.01},
        
        # 测试不同的探索率衰减
        {'n_bins': 10, 'alpha': 0.1, 'gamma': 0.99, 'epsilon_start': 0.5, 'epsilon_decay': 0.98, 'epsilon_end': 0.01},
        {'n_bins': 10, 'alpha': 0.1, 'gamma': 0.99, 'epsilon_start': 0.5, 'epsilon_decay': 0.995, 'epsilon_end': 0.01},
        
        # 测试不同的状态离散化粒度
        {'n_bins': 8, 'alpha': 0.1, 'gamma': 0.99, 'epsilon_start': 0.5, 'epsilon_decay': 0.99, 'epsilon_end': 0.01},
        {'n_bins': 12, 'alpha': 0.1, 'gamma': 0.99, 'epsilon_start': 0.5, 'epsilon_decay': 0.99, 'epsilon_end': 0.01},
        
        # 一些组合参数设置
        {'n_bins': 12, 'alpha': 0.1, 'gamma': 0.99, 'epsilon_start': 0.3, 'epsilon_decay': 0.995, 'epsilon_end': 0.01},
        {'n_bins': 10, 'alpha': 0.2, 'gamma': 0.99, 'epsilon_start': 0.3, 'epsilon_decay': 0.99, 'epsilon_end': 0.01},
    ]
    
    print(f"将测试 {len(param_sets)} 种参数组合")
    
    results = []
    
    for i, params in enumerate(param_sets):
        print(f"\n测试参数组合 {i+1}/{len(param_sets)}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # 为每组参数训练多次以减少随机性影响
        n_trials = 2
        trial_results = []
        
        start_time = time.time()
        for trial in range(n_trials):
            print(f"  试验 {trial+1}/{n_trials}...")
            result = train_once(params)
            trial_results.append(result)
        
        # 平均多次训练的结果
        avg_result = {
            'params': params,
            'avg_reward': np.mean([r['avg_reward'] for r in trial_results]),
            'last_100_avg': np.mean([r['last_100_avg'] for r in trial_results]),
            'convergence_episode': np.mean([r['convergence_episode'] for r in trial_results]),
            'stability': np.mean([r['stability'] for r in trial_results])
        }
        
        results.append(avg_result)
        
        elapsed = time.time() - start_time
        print(f"耗时: {elapsed:.1f}秒")
        print(f"平均结果: 最后100回合平均奖励={avg_result['last_100_avg']:.2f}, " 
              f"收敛回合={avg_result['convergence_episode']:.1f}, "
              f"稳定性={avg_result['stability']:.2f}")
    
    # 根据性能排序
    results.sort(key=lambda x: (x['stability'], x['last_100_avg']), reverse=True)
    
    # 显示最佳参数
    best_result = results[0]
    print("\n最佳参数组合:")
    for key, value in best_result['params'].items():
        print(f"{key}: {value}")
    
    print(f"\n性能指标:")
    print(f"最后100回合平均奖励: {best_result['last_100_avg']:.2f}")
    print(f"收敛回合: {best_result['convergence_episode']:.1f}")
    print(f"稳定性: {best_result['stability']:.2f}")
    
    return best_result, results

# 绘制结果柱状图
def plot_comparison(results):
    # 准备数据
    labels = [f"组合{i+1}" for i in range(len(results))]
    rewards = [r['last_100_avg'] for r in results]
    stability = [r['stability'] for r in results]
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 奖励柱状图
    plt.subplot(1, 2, 1)
    plt.bar(labels, rewards, color='skyblue')
    plt.xlabel('参数组合')
    plt.ylabel('平均奖励')
    plt.title('各参数组合的平均奖励')
    plt.xticks(rotation=45)
    
    # 稳定性柱状图
    plt.subplot(1, 2, 2)
    plt.bar(labels, stability, color='lightgreen')
    plt.xlabel('参数组合')
    plt.ylabel('稳定性')
    plt.title('各参数组合的稳定性')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('parameter_comparison.png')

# 使用最佳参数训练并可视化结果
def train_with_best_params(best_params, episodes=300):
    print("\n使用最佳参数进行训练...")
    
    # 设置参数
    params = best_params.copy()
    
    # 训练并评估 - 先不可视化
    print("进行训练中...")
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
    plt.title(f'最佳参数组合的学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('best_params_learning_curve.png')
    
    # 展示最后10回合的性能
    print(f"最后10回合平均奖励: {np.mean(result['episode_rewards'][-10:]):.2f}")
    
    # 最后在可视化环境中展示
    print("\n在可视化环境中展示最终效果...")
    env = gym.make('CartPole-v1', render_mode='human')
    
    # 创建并使用训练好的Q表
    n_dims = 4
    n_bins = params['n_bins']
    state_bounds = [
        (-2.4, 2.4),       # 小车位置范围
        (-4.0, 4.0),       # 小车速度范围
        (-0.2095, 0.2095), # 杆子角度范围
        (-4.0, 4.0)        # 杆子角速度范围
    ]
    
    q_table = np.zeros([n_bins] * n_dims + [env.action_space.n])
    
    # 用最佳参数重新训练一个Q表（不可视化）
    for episode in range(300):
        state, _ = env.reset(seed=episode)  # 使用不同的种子
        state = discretize_state(state, n_bins, state_bounds)
        
        epsilon = max(params['epsilon_end'], params['epsilon_start'] * (params['epsilon_decay'] ** episode))
        
        for step in range(500):
            action = select_action(state, q_table, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, n_bins, state_bounds)
            
            # Q-learning更新
            old_value = q_table[state + (action,)]
            next_best = np.max(q_table[next_state])
            td_target = reward + params['gamma'] * next_best
            td_error = td_target - old_value
            new_value = old_value + params['alpha'] * td_error
            q_table[state + (action,)] = new_value
            
            state = next_state
            
            if terminated or truncated:
                break
                
        if episode % 20 == 0:
            print(f"预训练回合 {episode+1}/300")
    
    # 展示视觉效果
    print("\n展示训练效果...")
    for episode in range(5):
        state, _ = env.reset(seed=1000+episode)  # 使用不同的种子
        state = discretize_state(state, n_bins, state_bounds)
        total_reward = 0
        
        for step in range(500):
            # 使用完全贪婪策略（不再探索）
            action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, n_bins, state_bounds)
            total_reward += reward
            
            state = next_state
            
            # 控制显示速度
            import time
            time.sleep(0.01)
            
            if terminated or truncated:
                break
                
        print(f"测试回合 {episode+1}/5, 总奖励: {total_reward}")
    
    env.close()
    
    return result

# 主函数
def main():
    print("==== CartPole最优超参数查找 ====")
    print("通过测试不同的超参数组合，寻找最优配置...\n")
    
    # 运行优化
    best_result, all_results = simple_optimization()
    
    # 绘制比较图
    plot_comparison(all_results)
    
    # 使用最佳参数训练并展示
    train_with_best_params(best_result['params'])
    
    print("\n==== 最终推荐的超参数组合 ====")
    for key, value in best_result['params'].items():
        print(f"{key}: {value}")
    
    print("\n超参数优化完成！")
    print("最佳参数组合已确定，学习曲线已绘制。")

if __name__ == "__main__":
    main() 