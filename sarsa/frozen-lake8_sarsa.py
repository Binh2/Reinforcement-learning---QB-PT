# QB - Frozen lake environment (No slipery) - SARSA
import matplotlib.pyplot as plt
import numpy as np
import time
import gym

# TRAI = 0
# XUONG = 1
# PHAI = 2
# LEN = 3

def choose_action(state, epsilon):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(qtable[state, :])
    return action
 

def update(state, next_state, reward, action, next_action):
    predict = qtable[state, action]
    target = reward + gamma * qtable[next_state, next_action]
    qtable[state, action] = qtable[state, action] + alpha * (target - predict)


def run_episodes(env, qtable, total_episodes, alpha, epsilon_start, epsilon_decay, epsilon_end, gamma):
    rewards = []
    epsilon = epsilon_start
    for episode in range(total_episodes):
        t = 0
        state = env.reset()
        action = choose_action(state, epsilon)
        while t < max_steps_per_episode:
            
            next_state, reward, done, info = env.step(action)
            next_action = choose_action(next_state, epsilon)
            update(state, next_state, reward, action, next_action)

            state = next_state
            action = next_action
            
            t += 1

            if done:
                break
        rewards.append(reward)
        epsilon = max(epsilon - epsilon_decay, epsilon_end)
    return rewards


env = gym.make('FrozenLake-v1', is_slippery=False)
epsilon_start = 0.9
epsilon_decay = 0.001
epsilon_end = 0
total_episodes = 1000
max_steps_per_episode = 100
alpha = 0.85
gamma = 0.95
qtable = np.zeros((env.observation_space.n, env.action_space.n))

# Huấn luyện máy
print("Đang huấn luyện...")
rewards = run_episodes(env, qtable, total_episodes, alpha, epsilon_start, epsilon_decay, epsilon_end, gamma)
print("Đã huấn luyện xong.")
print('Q-table = ', qtable, sep="\n")

# Lưu lại qtable để lần khác khỏi phải huấn luyện
np.save("qtable.npz", qtable)

# Tải lại qtable nếu đã lưu để đỡ phải huấn luyện
# qtable = np.load("qtable.npz")

# Hiển thị đồ thị thể hiện phần thưởng nhận được trong c1000 tập huấn luyện
fig, ax = plt.subplots()
ax.scatter([i for i in range(total_episodes)], rewards, s=0.1)
ax.set_xlabel("Tập huấn luyện thứ")
ax.set_ylabel("PHần thưởng nhận được tại tập huấn luyện tương ứng")
plt.show()

# input("Nhấn enter để mở animation.")
epsilon = 0
state = env.reset()
env.render()
for i in range(max_steps_per_episode):
    action = choose_action(state, epsilon)
    print(action, end=" ")
    state, reward, done, info = env.step(action)
    
    env.render()
    time.sleep(0.8)
    
    if done:
        break

print()
print("Chiến thắng!!!")