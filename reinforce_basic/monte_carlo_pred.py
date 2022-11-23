import numpy as np
from gridworld_env import Env


class monte_carlo_pred:
    def __init__(self):
        self.value_table = np.zeros((Env.HEIGHT, Env.WIDTH))
        self.env = Env()
        self.discount_factor = 0.9
        self.epsilon = 0.5
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.05
        self.learning_rate = 0.1
        self.samples = []

    def get_action(self):
        possible_actions = self.env.get_possible_actions()

        if np.random.rand() < self.epsilon: # 탐험 수행
            return np.random.choice(possible_actions)

        values = []
        for action in possible_actions:
            state = self.env.get_player_state()
            if action == Env.UP:
                state[0] -= 1
            if action == Env.DOWN:
                state[0] += 1
            if action == Env.LEFT:
                state[1] -= 1
            if action == Env.RIGHT:
                state[1] += 1
            values.append(self.value_table[tuple(state)])

        values = np.array(values)
        max_actions = np.argwhere(values == np.max(values)) # 2차원 배열로 반환 ex)[[0],[1]]
        max_actions = max_actions.reshape(-1)# 반환된 값을 1차원 배열로 변환 해야함
        action_idx = np.random.choice(max_actions)
        return possible_actions[action_idx]

    def save_sample(self, reward, state_codi):
        self.samples.append((reward, state_codi))

    def update_value(self):
        # v(s) <= v(s) + a(G_n - v(s))
        G_t = 0
        rwd_sum = 0
        for (reward, state) in reversed(self.samples):
            v_t = self.value_table[state]
            G_t = reward + self.discount_factor*G_t
            v_t = v_t + self.learning_rate*(G_t-v_t)
            self.value_table[state] = v_t
            rwd_sum += reward


    def run(self):
        episode_cnt = 100
        for ep in range(episode_cnt):

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            while True:
                action = self.get_action()
                reward, state_codi, done = self.env.move(action)
                self.save_sample(reward, state_codi)
                if done:
                    self.update_value()
                    print("\n")
                    print(f"episode: {ep}")
                    print(f"move_cnt: {len(self.samples)}")
                    print(self.value_table)
                    self.samples.clear()
                    self.env.reset()
                    break

if __name__ == "__main__":
    monte = monte_carlo_pred()
    monte.run()