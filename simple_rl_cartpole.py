import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# 创建CartPole环境 - 修改渲染模式
env = gym.make('CartPole-v1', render_mode='human')

# 确保环境正确渲染的函数
def test_environment():
    print("测试CartPole环境...")
    test_env = gym.make('CartPole-v1', render_mode='human')
    state, _ = test_env.reset()
    print("环境状态维度:", len(state))
    print("初始状态:", state)
    print("渲染环境中...")
    test_env.render()
    sleep(3)  # 暂停3秒钟查看环境
    print("执行一些随机动作...")
    for _ in range(20):
        action = test_env.action_space.sample()
        _, _, terminated, truncated, _ = test_env.step(action)
        test_env.render()
        if terminated or truncated:
            break
        sleep(0.1)
    test_env.close()
    print("环境测试完成")

# 运行环境测试
test_environment()

# 离散化状态空间
n_bins = 10  # 每个维度的箱子数量
n_dims = 4   # 状态空间的维度数量

# 定义Q表的形状: (状态空间的每个维度的箱子数量) x (动作数量)
q_table = np.zeros([n_bins] * n_dims + [env.action_space.n])

# 定义学习参数
alpha = 0.1      # 学习率
gamma = 0.99     # 折扣因子
epsilon_start = 0.5   # 初始探索率
epsilon_end = 0.01    # 最终探索率
epsilon_decay = 0.99  # 探索率衰减因子

# 离散化状态的函数
def discretize_state(state):
    # 定义每个维度的范围
    cart_pos_bins = np.linspace(-2.4, 2.4, n_bins)
    cart_vel_bins = np.linspace(-4, 4, n_bins)
    pole_ang_bins = np.linspace(-0.2095, 0.2095, n_bins)
    pole_vel_bins = np.linspace(-4, 4, n_bins)
    
    # 将连续状态转为离散索引
    cart_pos_idx = np.digitize(state[0], cart_pos_bins)
    cart_vel_idx = np.digitize(state[1], cart_vel_bins)
    pole_ang_idx = np.digitize(state[2], pole_ang_bins)
    pole_vel_idx = np.digitize(state[3], pole_vel_bins)
    
    # 确保索引在有效范围内
    cart_pos_idx = min(cart_pos_idx, n_bins - 1)
    cart_vel_idx = min(cart_vel_idx, n_bins - 1)
    pole_ang_idx = min(pole_ang_idx, n_bins - 1)
    pole_vel_idx = min(pole_vel_idx, n_bins - 1)
    
    return (cart_pos_idx, cart_vel_idx, pole_ang_idx, pole_vel_idx)

# 选择动作的函数
def select_action(state, epsilon):
    # epsilon-greedy策略
    if np.random.random() < epsilon:
        return env.action_space.sample()  # 随机探索
    else:
        return np.argmax(q_table[state])  # 选择最优动作

# 训练参数
n_episodes = 200  # 增加到200回合
max_steps = 200

# 训练过程
rewards = []

print("开始训练...")
for episode in range(n_episodes):
    # 重置环境
    state, _ = env.reset()
    state = discretize_state(state)
    total_reward = 0
    
    # 计算当前回合的探索率
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
    
    for step in range(max_steps):
        # 选择动作
        action = select_action(state, epsilon)
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_state)
        total_reward += reward
        
        # 更新Q表
        old_value = q_table[state + (action,)]
        next_best = np.max(q_table[next_state])
        
        # Q-learning更新公式
        new_value = old_value + alpha * (reward + gamma * next_best - old_value)
        q_table[state + (action,)] = new_value
        
        # 状态转移
        state = next_state
        
        # 如果回合结束，跳出循环
        if done:
            break
        
        # 为了看清训练过程，可以添加一点延迟
        sleep(0.01)
    
    # 记录每个回合的总奖励
    rewards.append(total_reward)
    print(f"回合 {episode+1}/{n_episodes}, 总奖励: {total_reward}, 探索率: {epsilon:.4f}")

# 关闭环境
env.close()

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(rewards, alpha=0.6, label='原始奖励')

# 添加滑动平均线以更好地观察趋势
window_size = 10
rewards_smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(window_size-1, len(rewards)), rewards_smoothed, linewidth=2, label=f'{window_size}回合滑动平均')

plt.xlabel('回合')
plt.ylabel('总奖励')
plt.title('Q-learning在CartPole环境中的学习曲线 (200回合)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('learning_curve_200.png')
plt.show()

print("训练完成！学习曲线已保存为'learning_curve_200.png'") 