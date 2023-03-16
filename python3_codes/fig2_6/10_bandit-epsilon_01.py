import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

cmap = cm.get_cmap("Spectral")

summation = 0
iterate = 0
Nexp = 3
Npulls = 150


fig = plt.figure(figsize=(4, 9))
fig.patch.set_facecolor("white")


class Bandit:
    def __init__(self, N_arm):
        self.N_arm = N_arm
        self.arm_values = np.random.normal(0, 1, self.N_arm)
        self.K = np.zeros(self.N_arm)  # i 番目のアームを引いた回数
        self.est_values = np.ones(self.N_arm) * 0

    def get_reward(self, action):
        noise = np.random.normal(0, 1)
        reward = self.arm_values[action] + noise
        return reward

    def choose_eps_greedy(self, epsilon):
        p = np.random.random()
        if epsilon > p:
            return np.random.randint(self.N_arm)  # ランダムにアームを選ぶ
        else:
            return np.argmax(self.est_values)  # 現状維持、これまでの最大値を選択

    # 価値の更新
    def update_est(self, action, reward):
        self.K[action] += 1
        # 価値 +=（報酬-価値/これまでにi 番目のアームを選択した回数）
        alpha = 1.0 / self.K[action]
        self.est_values[action] += alpha * (reward - self.est_values[action])
        return self.est_values[action]


def sets_color(l):
    if l < 0:
        return "r"  # red
    else:
        return "b"  # blue


def experiment(bandit, Npulls, epsilon):
    history = []
    global summation, iterate
    iterate = iterate + 1
    set_color = ["r", "b", "g", "c", "m", "y", "k", "orange", "r", "b"]  # アーム分色を準備
    for i in range(Npulls):
        action = bandit.choose_eps_greedy(epsilon)
        R = bandit.get_reward(action)
        bandit.update_est(action, R)
        history.append(R)

        plt.subplot(311)
        plt.scatter(i, bandit.update_est(action, R), c=set_color[action], alpha=1, s=20)
        plt.pause(0.001)

        plt.subplot(312)
        plt.scatter(i, bandit.get_reward(action), c=set_color[action], alpha=1, s=20)

        plt.subplot(313)
        summation[i] = summation[i] + history[i]

        plt.scatter(
            Npulls * (iterate - 1) + i,
            summation[i] / (15.0),
            c=sets_color(summation[i]),
            alpha=0.8,
            s=20,
        )
        plt.pause(0.001)

    return np.array(history)


summation = np.zeros(Npulls)
avg_outcome_eps0p1 = np.zeros(Npulls)

for i in range(Nexp):
    bandit = Bandit(10)

    plt.clf()
    avg_outcome_eps0p1 += experiment(bandit, Npulls, 0.1)

plt.show()
avg_outcome_eps0p1 /= np.float(Nexp)
fig = plt.figure()
fig.patch.set_facecolor("white")
plt.plot(avg_outcome_eps0p1, label="eps = 0.1")
plt.legend()
plt.show()
