import numpy as np
from gridworld_env import Env
from collections import defaultdict


class Sarsa:
    def __init__(self):
        self.qvalue_table = defaultdict(lambda : [0.0]*Env.ACTION_SIZE)
        self.env = Env()
        self.discount_factor = 0.9
        self.epsilon = 0.5
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.05
        self.learning_rate = 0.1
        self.samples = []

    def get_action(self, state):
        possible_actions = self.env.get_possible_actions(state)

        if np.random.rand() < self.epsilon: # 탐험 수행
            return np.random.choice(possible_actions)

        values = []
        q_values = self.qvalue_table[tuple(state)]
        for action in possible_actions:
            values.append(q_values[action])

        values = np.array(values)
        max_actions = np.argwhere(values == np.max(values)) # np.argwhere는 2차원 배열로 반환 ex)[[0],[1]]
        max_actions = max_actions.reshape(-1)# 반환된 값을 1차원 배열로 변환 해야함
        action_idx = np.random.choice(max_actions)
        return possible_actions[action_idx]

    def learn(self, s0, a0, r1, s1, a1):
        s0qval = self.qvalue_table[tuple(s0)][a0]
        s1qval = self.qvalue_table[tuple(s1)][a1]
        next_s0value = s0qval + self.learning_rate*(r1+self.discount_factor*s1qval - s0qval)
        self.qvalue_table[tuple(s0)][a0] = next_s0value

    def print_qvalue_map(self):
        pass

    def run(self):
        episode_cnt = 100
        for ep in range(episode_cnt):

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            state = self.env.get_player_state()
            action = self.get_action(state)
            move_cnt = 0
            while True:
                reward, next_state, done = self.env.move(action)
                next_action = self.get_action(next_state)
                self.learn(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                move_cnt += 1
                if done:
                    print("\n")
                    print(f"episode: {ep}")
                    print(f"move_cnt: {move_cnt}, epsilon:{self.epsilon:.4f}")
                    #print(self.value_table)
                    self.env.reset()
                    break

if __name__ == "__main__":
    sarsa = Sarsa()
    sarsa.run()