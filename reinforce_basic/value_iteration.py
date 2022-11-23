import numpy as np
from typing import Final

class Env:
    WIDTH: Final = 4
    HEIGHT: Final = 4

    LEFT:Final = 0
    RIGHT: Final = 1
    UP: Final = 2
    DOWN: Final = 3

    ACTION_SIZE:Final = 4
    Discount_factor:Final = 0.9

    def __init__(self):
        self.reward_table = np.zeros((self.HEIGHT, self.WIDTH))
        self.reward_table[0, 0] = 1
        self.reward_table[1, 2] = -1
        self.reward_table[2, 2] = -1
    def get_reward(self, state):
        return self.reward_table[state]

class value_iteration:
    LEFT, RIGHT, UP, DOWN = range(4)
    def __init__(self):
        self.value_table = np.zeros((Env.HEIGHT, Env.WIDTH))
        self.env = Env()
        #print(self.policy_table)

    def value_iteration(self):
        next_value_table = np.zeros_like(self.value_table)
        for row in range(Env.HEIGHT):
            for col in range(Env.WIDTH):
                next_state_vals = np.zeros(Env.ACTION_SIZE)
                next_state_rwds = np.zeros(Env.ACTION_SIZE)

                # 상단 상태의 가치
                if row > 0:
                    next_state_vals[Env.UP] = self.value_table[row - 1, col]
                    next_state_rwds[Env.UP] = self.env.get_reward((row - 1, col))
                # 하단 상태의 가치
                if row < Env.HEIGHT - 1:
                    next_state_vals[Env.DOWN] = self.value_table[row + 1, col]
                    next_state_rwds[Env.DOWN] = self.env.get_reward((row + 1, col))
                # 좌측 상태의 가치
                if col > 0:
                    next_state_vals[Env.LEFT] = self.value_table[row, col - 1]
                    next_state_rwds[Env.LEFT] = self.env.get_reward((row, col - 1))
                # 우측 상태의 가치
                if col < Env.WIDTH - 1:
                    next_state_vals[Env.RIGHT] = self.value_table[row, col + 1]
                    next_state_rwds[Env.RIGHT] = self.env.get_reward((row, col + 1))

                values = np.zeros(Env.ACTION_SIZE)

                for i in range(Env.ACTION_SIZE):
                    values[i] = (next_state_rwds[i] + Env.Discount_factor * next_state_vals[i])

                next_value_table[row, col] = np.max(values)

        self.value_table = next_value_table



    def run(self):
        loop_max = 500
        for loop_idx in range(loop_max):
            self.value_iteration()
            print(f"Loop: {loop_idx + 1}")
            print(self.value_table)


if __name__ == "__main__":
    a = 0
    #state_value_fun()
    value = value_iteration()
    value.run()



