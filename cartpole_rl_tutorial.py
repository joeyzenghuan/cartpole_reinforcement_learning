"""
强化学习入门教程：使用Q-learning算法训练CartPole平衡问题
==========================================================

这是一个面向新手的强化学习入门教程，使用经典的CartPole问题作为例子。
CartPole是一个小车平衡杆的问题：一个小车上面立着一根杆子，
目标是通过向左或向右移动小车来保持杆子平衡不倒下。

这个程序使用Q-learning（一种基础的强化学习算法）来训练一个智能体，
学习如何在CartPole环境中保持平衡。

作者：AI助手
日期：2023
"""

# 导入必要的库
import gymnasium as gym  # 强化学习环境库（新版本的gym）
import numpy as np      # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
from time import sleep  # 控制显示速度

# 第一步：创建CartPole环境
# ======================
# render_mode='human'表示我们希望看到可视化的训练过程
env = gym.make('CartPole-v1', render_mode='human')

# 第二步：定义状态空间的离散化参数
# ============================
# 由于Q-learning通常用于离散状态空间，而CartPole的状态是连续的，
# 我们需要将连续状态离散化为有限数量的"箱子"
n_bins = 10  # 每个状态维度的箱子数量
n_dims = 4   # 状态空间的维度（小车位置、速度、杆子角度、角速度）

# 第三步：创建Q表
# =============
# Q表存储每个状态-动作对的价值估计
# 形状为[10,10,10,10,2]，共有10^4*2=20,000个值

# 1. [n_bins] * n_dims 创建一个列表，其中包含n_dims个n_bins值。因为n_bins=10且n_dims=4，所以这一部分等同于[10, 10, 10, 10]。
# 2. + [env.action_space.n] 将动作空间的大小添加到列表末尾。在CartPole环境中，动作空间大小为2（向左或向右）。所以整个列表变成[10, 10, 10, 10, 2]。
# np.zeros() 创建一个形状为[10, 10, 10, 10, 2]的多维数组，所有元素初始化为0。
q_table = np.zeros([n_bins] * n_dims + [env.action_space.n])

# 第四步：定义学习参数
# =================
alpha = 0.1        # 学习率：控制新信息更新Q值的程度
gamma = 0.99       # 折扣因子：控制未来奖励的重要性
epsilon_start = 0.5  # 初始探索率：刚开始时随机探索的概率
epsilon_end = 0.01   # 最终探索率：训练后期的最小探索概率
epsilon_decay = 0.99 # 探索率衰减因子：控制探索率下降的速度

# 第五步：定义状态离散化函数
# =======================
def discretize_state(state):
    """
    将连续的状态值转换为离散的箱子索引
    
    参数:
        state (numpy.array): 包含4个浮点数的数组，表示环境的当前状态
                            [小车位置, 小车速度, 杆子角度, 杆子角速度]
    
    返回:
        tuple: 包含4个整数的元组，表示离散化后的状态索引
    """
    # 为每个状态维度定义值域范围
    cart_pos_bins = np.linspace(-2.4, 2.4, n_bins)       # 小车位置范围
    cart_vel_bins = np.linspace(-4, 4, n_bins)           # 小车速度范围
    pole_ang_bins = np.linspace(-0.2095, 0.2095, n_bins) # 杆子角度范围（约±12度）
    pole_vel_bins = np.linspace(-4, 4, n_bins)           # 杆子角速度范围
    
    # 使用np.digitize将连续值映射到箱子索引
    # 例如，如果小车位置是1.5，它可能被映射到索引7
    cart_pos_idx = np.digitize(state[0], cart_pos_bins)
    cart_vel_idx = np.digitize(state[1], cart_vel_bins)
    pole_ang_idx = np.digitize(state[2], pole_ang_bins)
    pole_vel_idx = np.digitize(state[3], pole_vel_bins)
    
    # 确保索引不会超出有效范围（0-9）
    cart_pos_idx = min(cart_pos_idx, n_bins - 1)
    cart_vel_idx = min(cart_vel_idx, n_bins - 1)
    pole_ang_idx = min(pole_ang_idx, n_bins - 1)
    pole_vel_idx = min(pole_vel_idx, n_bins - 1)
    
    # 返回离散化的状态索引元组
    return (cart_pos_idx, cart_vel_idx, pole_ang_idx, pole_vel_idx)

# 第六步：定义动作选择函数
# ======================
def select_action(state, epsilon):
    """
    使用epsilon-greedy策略选择动作
    
    参数:
        state (tuple): 离散化的状态索引
        epsilon (float): 当前的探索率 (0-1之间)
    
    返回:
        int: 选择的动作 (0表示向左, 1表示向右)
    """
    # 生成一个随机数，用于决定是探索还是利用
    if np.random.random() < epsilon:
        # 探索：随机选择一个动作
        return env.action_space.sample()
    else:                                                                                                                                                                                                                                                                                                                
        # 利用：选择当前状态下价值最高的动作
        return np.argmax(q_table[state])

# 第七步：设置训练参数
# =================
n_episodes = 1000  # 训练的总回合数
max_steps = 200   # 每个回合最大步数（CartPole-v1中为500，这里我们限制为200以加快训练）

# 训练速度控制参数
# visualization_mode = 'human'  # 可视化模式: 'human'(实时显示), 'rgb_array'(仅保存), 'none'(无可视化)
visualization_mode = 'none'  # 可视化模式: 'human'(实时显示), 'rgb_array'(仅保存), 'none'(无可视化)
# render_delay = 0.01  # 每步渲染后的延迟时间(秒): 0表示最快速度，值越大越慢
render_delay = 0  # 每步渲染后的延迟时间(秒): 0表示最快速度，值越大越慢
render_freq = 1      # 渲染频率: 每隔多少步显示一次，1表示每步都显示，10表示每10步显示一次
episode_render_freq = 1  # 每隔多少回合显示一次: 1表示每个回合都显示，5表示每5个回合显示一次

# 根据可视化模式创建环境
if visualization_mode == 'none':
    env = gym.make('CartPole-v1', render_mode=None)  # 无可视化，训练最快
elif visualization_mode == 'rgb_array':
    env = gym.make('CartPole-v1', render_mode='rgb_array')  # 离线渲染，训练较快
else:
    env = gym.make('CartPole-v1', render_mode='human')  # 实时显示，训练较慢

# 第八步：开始训练过程
# =================
# 用于记录每个回合的总奖励
rewards = []

print("开始强化学习训练...")
print("环境信息:")
print(f"  状态空间: {env.observation_space}")
print(f"  动作空间: {env.action_space}")
print(f"  奖励机制: 每保持平衡一步 +1分")
print(f"  终止条件: 杆子倾斜过大(>12度)或小车移出边界(>2.4单位)")
print(f"训练参数:")
print(f"  回合数: {n_episodes}")
print(f"  学习率: {alpha}")
print(f"  折扣因子: {gamma}")
print(f"  初始探索率: {epsilon_start}")
print(f"  探索率衰减: {epsilon_decay}")
print(f"速度设置:")
print(f"  可视化模式: {visualization_mode}")
print(f"  渲染延迟: {render_delay}秒/步")
print(f"  渲染频率: 每{render_freq}步")
print(f"  回合显示频率: 每{episode_render_freq}回合")
print("==========================")

for episode in range(n_episodes):
    # 重置环境，获取初始状态
    state, _ = env.reset()
    # 将连续状态离散化
    state = discretize_state(state)
    total_reward = 0
    
    # 计算当前回合的探索率（随训练进行逐渐减小）
    # 减小探索率意味着智能体会越来越倾向于利用已有知识而非探索新动作
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
    
    # 决定是否显示当前回合
    should_render_episode = (episode % episode_render_freq == 0) and (visualization_mode != 'none')
    
    # 在当前回合中执行步骤
    for step in range(max_steps):
        #################################################################
        # 第一步：选择动作（行动）
        #################################################################
        # 基于当前状态和探索率选择动作
        # 这里使用epsilon-greedy策略：有epsilon的概率随机探索，1-epsilon的概率选择当前认为最好的动作
        action = select_action(state, epsilon)
        
        #################################################################
        # 第二步：执行动作，与环境互动
        #################################################################
        # env.step(action)执行动作并返回：
        # - next_state: 执行动作后的新状态（小车新位置、速度、杆子新角度等）
        # - reward: 获得的奖励（在CartPole中，每步维持平衡得到+1分）
        # - terminated: 是否因为失败而结束（杆子倒了或小车出界）
        # - truncated: 是否因为达到最大步数而结束
        # - _: 其他信息（此处不使用）
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # 判断当前回合是否结束
        done = terminated or truncated
        
        # 将连续的新状态转换为离散状态，便于查询Q表
        next_state = discretize_state(next_state)
        
        # 累加当前回合的总奖励
        total_reward += reward
        
        #################################################################
        # 第三步：Q-learning的核心 - 更新知识（Q表）
        #################################################################
        # 这是Q-learning算法最关键的部分，通过经验更新Q表中的值
        
        # 更新公式: Q(s,a) = Q(s,a) + α[r + γ·maxQ(s',a') - Q(s,a)]
        # 白话解释：（有点像大模型中，更新每个token的概率分布，让对的token概率更高）
        # "新的价值估计 = 旧的价值估计 + 学习率 * (实际获得的价值 - 之前估计的价值)"
        # 其中"实际获得的价值"= 即时奖励 + 折扣因子 * 下一状态的最佳行动价值
        
        # 步骤1：获取当前状态-动作对在Q表中的当前价值
        old_value = q_table[state + (action,)]
        # state + (action,) 是将状态元组和动作合并，形成Q表的索引
        # 例如，如果state=(3,4,2,6)且action=1，则索引为(3,4,2,6,1)
        
        # 步骤2：找出下一个状态中所有可能动作的最大价值
        next_best = np.max(q_table[next_state])
        # 这体现了"贪婪"策略，假设在下一状态会选择价值最高的动作
        
        # 步骤3：计算"时间差分目标"(TD target)，即实际价值估计
        # 时间差分目标 = 即时奖励 + 折扣因子 * 下一状态的最大价值
        td_target = reward + gamma * next_best
        
        # 步骤4：计算"时间差分误差"(TD error)，即估计与实际的差距
        # 误差 = 时间差分目标 - 当前估计
        td_error = td_target - old_value
        
        # 步骤5：更新Q表中对应值
        # 新价值 = 旧价值 + 学习率 * 误差
        # α(alpha)控制更新的幅度，较小的值使学习更平稳但更慢
        new_value = old_value + alpha * td_error
        
        # 步骤6：将新的价值写回Q表对应位置
        q_table[state + (action,)] = new_value
        
        # 直观解释：
        # 1. 如果结果比预期好（正误差），增加这个状态-动作对的价值
        # 2. 如果结果比预期差（负误差），减少这个状态-动作对的价值
        # 3. 通过多次尝试，Q表将逐渐收敛到真实的价值
        
        #################################################################
        # 第四步：状态更新，准备下一步决策
        #################################################################
        # 状态转移到下一个状态，继续下一步决策循环
        state = next_state
        
        # 如果回合结束（杆子倒下或小车出界），提前退出循环
        if done:
            break
        
        # 控制渲染和延迟
        if should_render_episode and step % render_freq == 0:
            if visualization_mode == 'human':
                # 为了便于观察训练过程，添加轻微的延迟
                sleep(render_delay)
            elif visualization_mode == 'rgb_array' and episode % 50 == 0 and step == 0:
                # 每50个回合保存一个截图
                img = env.render()
                plt.imsave(f'episode_{episode}_step_{step}.png', img)
    
    # 记录当前回合的总奖励
    rewards.append(total_reward)
    # print(f"回合 {episode+1}/{n_episodes}, 总奖励: {total_reward}, 探索率: {epsilon:.4f}")

# 第九步：关闭环境
# =============
env.close()

# 打印Q表中的一部分，帮助理解模型学到了什么
print("\n===== Q表内容分析 =====")
print("Q表形状:", q_table.shape)
print("Q表中非零值的数量:", np.count_nonzero(q_table), "占总参数的", 
      round(100 * np.count_nonzero(q_table) / q_table.size, 2), "%")

# 找出Q表中价值最高的几个状态-动作对
flat_indices = np.argsort(q_table.flatten())[-20:]  # 获取最大的20个值的索引
high_values = []

print("\n最有价值的20个状态-动作对:")
print("序号 | 小车位置 | 小车速度 | 杆子角度 | 杆子角速度 | 动作 | Q值")
print("-" * 70)

for i, flat_idx in enumerate(flat_indices[::-1], 1):  # 从最大值开始倒序
    # 将扁平索引转换为多维索引
    indices = np.unravel_index(flat_idx, q_table.shape)
    value = q_table[indices]
    
    # 获取对应的状态和动作
    cart_pos, cart_vel, pole_ang, pole_vel, action = indices
    
    # 将离散索引转换回近似的连续值，以便更好理解
    cart_pos_value = -2.4 + (cart_pos * 4.8 / (n_bins-1))
    cart_vel_value = -4.0 + (cart_vel * 8.0 / (n_bins-1))
    pole_ang_value = -0.2095 + (pole_ang * 0.419 / (n_bins-1))
    pole_vel_value = -4.0 + (pole_vel * 8.0 / (n_bins-1))
    
    # 动作描述
    action_desc = "左" if action == 0 else "右"
    
    print(f"{i:2d} | {cart_pos_value:+.2f} | {cart_vel_value:+.2f} | {pole_ang_value:+.2f} | {pole_vel_value:+.2f} | {action_desc} | {value:.2f}")
    
    high_values.append((cart_pos_value, cart_vel_value, pole_ang_value, pole_vel_value, action_desc, value))

# 分析最常见的策略模式
print("\n策略分析:")
# 分析杆子倾斜时的行动策略
left_tilt_right_action = 0
left_tilt_left_action = 0
right_tilt_right_action = 0
right_tilt_left_action = 0

# 遍历Q表中的一部分来分析策略
pole_mid_idx = n_bins // 2
sample_idxs = np.array(np.meshgrid(range(n_bins), range(n_bins), [pole_mid_idx-1, pole_mid_idx+1], range(n_bins))).T.reshape(-1, 4)

for cart_pos, cart_vel, pole_ang, pole_vel in sample_idxs:
    best_action = np.argmax(q_table[cart_pos, cart_vel, pole_ang, pole_vel])
    if pole_ang < pole_mid_idx:  # 杆子向左倾斜
        if best_action == 0:     # 向左动作
            left_tilt_left_action += 1
        else:                    # 向右动作
            left_tilt_right_action += 1
    else:                        # 杆子向右倾斜
        if best_action == 0:     # 向左动作
            right_tilt_left_action += 1
        else:                    # 向右动作
            right_tilt_right_action += 1

print(f"杆子向左倾斜时选择向左移动的比例: {left_tilt_left_action/(left_tilt_left_action+left_tilt_right_action):.2f}")
print(f"杆子向右倾斜时选择向右移动的比例: {right_tilt_right_action/(right_tilt_right_action+right_tilt_left_action):.2f}")

print("\n简单解释:")
if left_tilt_left_action > left_tilt_right_action and right_tilt_right_action > right_tilt_left_action:
    print("模型学习到了正确的平衡策略：当杆子向左倾斜时，小车向左移动；当杆子向右倾斜时，小车向右移动。")
else:
    print("模型可能尚未完全学习到最优策略，或者学习到了一种非直观但有效的策略。")

# 第十步：绘制学习曲线并分析训练结果
# =============================
plt.figure(figsize=(10, 6))
plt.plot(rewards, alpha=0.6, label='原始奖励')

# 添加滑动平均线，帮助观察整体趋势
window_size = 10
rewards_smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(window_size-1, len(rewards)), rewards_smoothed, linewidth=2, label=f'{window_size}回合滑动平均')

plt.xlabel('回合')
plt.ylabel('总奖励')
plt.title('Q-learning在CartPole环境中的学习曲线 (200回合)')
plt.legend()
plt.grid(True, alpha=0.3)
# plt.savefig('learning_curve_tutorial.png')
plt.show()

print("==== 训练完成 ====")
print(f"学习曲线已保存为'learning_curve_tutorial.png'")
print(f"最后10回合平均奖励: {np.mean(rewards[-10:]):.2f}")
print("\n强化学习关键概念:")
print("1. 状态-动作价值函数(Q函数): 存储在Q表中的每个状态-动作对的预期回报")
print("2. 探索与利用权衡: 通过epsilon-greedy策略平衡随机探索和利用已知的最佳动作")
print("3. 时序差分学习: 使用当前观察到的奖励和下一状态的估计来更新当前状态的价值")
print("4. 离散化: 将连续状态空间转换为离散表示，使得Q-learning可以应用")
print("\n进阶学习方向:")
print("1. 尝试深度Q网络(DQN)处理更复杂环境")
print("2. 探索基于策略的方法如策略梯度(Policy Gradient)")
print("3. 学习Actor-Critic架构结合价值和策略方法的优点") 