import numpy as np
from gridworld_env import Env
from collections import defaultdict


class Q_Learning:
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

    def learn(self, s0, a0, r1, s1):
        s0qval = self.qvalue_table[tuple(s0)][a0]
        max_s1qval = max(self.qvalue_table[tuple(s1)]) # 최적 방정식 사용
        next_s0value = s0qval + self.learning_rate*(r1+self.discount_factor*max_s1qval - s0qval)
        self.qvalue_table[tuple(s0)][a0] = next_s0value

    # 저장된 q값 중 최대 값에 해당하는 방향의 스테이지에 +1을 해줘 q 값을 통한 이동 경로를 보여줌
    def print_qvalue_map(self):
        qvalue_map = np.zeros((self.env.HEIGHT, self.env.WIDTH))
        for row in range(self.env.HEIGHT):
            for col in range(self.env.WIDTH):
                qvalue = self.qvalue_table[(row, col)]

                possible_act = self.env.get_possible_actions([row,col])
                max_acts = np.argwhere(qvalue == np.max(qvalue))
                max_acts = max_acts.reshape(-1)

                for action in possible_act:
                    if action in max_acts:
                        if action == self.env.UP:
                            qvalue_map[row-1, col] += 1
                        if action == self.env.DOWN:
                            qvalue_map[row+1, col] += 1
                        if action == self.env.LEFT:
                            qvalue_map[row, col-1] += 1
                        if action == self.env.RIGHT:
                            qvalue_map[row, col+1] += 1
        print(qvalue_map)

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
                self.learn(state, action, reward, next_state)
                state = next_state
                action = next_action
                move_cnt += 1
                if done:
                    print("\n")
                    print(f"episode: {ep}")
                    print(f"move_cnt: {move_cnt}, epsilon:{self.epsilon:.4f}")
                    self.print_qvalue_map()
                    self.env.reset()
                    break

if __name__ == "__main__":
    qlearning = Q_Learning()
    qlearning.run()