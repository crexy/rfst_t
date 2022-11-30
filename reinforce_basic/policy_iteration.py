import numpy as np
from typing import Final

def state_value_fun():
    value_table = np.zeros((4,4))
    #policy_table = np.zeros((4, 4, 4))

    reward_table = np.zeros((4,4))
    reward_table[0, 0] = 1
    reward_table[1, 2] = -1
    reward_table[2, 2] = -1

    LEFT, RIGHT, UP, DOWN = range(4)

    print(value_table)

    loop_max = 100
    discount_factor = 0.9

    for loop_idx in range(loop_max+1):
        next_value_table = np.zeros_like(value_table)

        row_len = value_table.shape[0]
        col_len = value_table.shape[1]
        for row in range(row_len):
            for col in range(col_len):

                next_vals = np.zeros(4)
                next_rwds = np.zeros(4)

                value_sum = 0.

                # 상단 상태의 가치
                if row > 0:
                    next_vals[UP] = value_table[row-1, col]
                    next_rwds[UP] = reward_table[row-1, col]
                # 하단 상태의 가치
                if row < row_len-1:
                    next_vals[DOWN] = value_table[row+1, col]
                    next_rwds[DOWN] = reward_table[row+1, col]
                # 좌측 상태의 가치
                if col > 0:
                    next_vals[LEFT] = value_table[row, col-1]
                    next_rwds[LEFT] = reward_table[row, col - 1]
                # 우측 상태의 가치
                if col < col_len-1:
                    next_vals[RIGHT] = value_table[row, col+1]
                    next_rwds[RIGHT] = reward_table[row, col + 1]

                for i in range(4):
                    next_value_table[row, col] += (next_rwds[i]+discount_factor*next_vals[i])*0.25

        value_table = next_value_table

        print(f"Loop: {loop_idx+1}")
        print(value_table)

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

class policy_iteration:
    LEFT, RIGHT, UP, DOWN = range(4)
    def __init__(self):
        self.value_table = np.zeros((Env.HEIGHT, Env.WIDTH))
        #self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * Env.width for _ in range(Env.height)]
        self.policy_table = np.ones((Env.HEIGHT, Env.HEIGHT, Env.ACTION_SIZE))
        self.policy_table *= 0.25
        self.env = Env()
        #print(self.policy_table)

    def policy_evaluation(self):
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

                for i in range(Env.ACTION_SIZE):
                    next_value_table[row, col] += (next_state_rwds[i] + Env.Discount_factor * next_state_vals[i]) * self.policy_table[row, col, i]

        self.value_table = next_value_table

    def policy_improvement(self):
        next_policy_table = np.zeros_like(self.policy_table)
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

                qvalues = np.zeros(Env.ACTION_SIZE)

                for i in range(Env.ACTION_SIZE):
                    qvalues[i] = (next_state_rwds[i] + Env.Discount_factor * next_state_vals[i]) * self.policy_table[row, col, i]

                max_idxes = np.argwhere(qvalues == np.max(qvalues))
                probabilities = 1/len(max_idxes)

                for idx in max_idxes:
                    next_policy_table[row, col, idx] = probabilities

        self.policy_table = next_policy_table

    def run(self):
        loop_max = 100
        for loop_idx in range(loop_max):
            self.policy_evaluation()
            self.policy_improvement()
            #action_table = np.argmax(self.policy_table, axis=2)
            print(f"Loop: {loop_idx + 1}")
            #print(action_table)
            print(self.value_table)


if __name__ == "__main__":
    a = 0
    #state_value_fun()
    policy = policy_iteration()
    policy.run()



