import numpy as np
from typing import Final

class Env:
    WIDTH: Final = 5
    HEIGHT: Final = 5

    LEFT: Final = 0
    RIGHT: Final = 1
    UP: Final = 2
    DOWN: Final = 3

    ACTION_SIZE:Final = 4

    def __init__(self):
        self.reward_table = np.zeros((self.HEIGHT, self.WIDTH))
        self.reward_table[1, 1] = 1
        #self.reward_table[0, 3] = 1
        self.reward_table[2, 2] = -1
        self.reward_table[2, 4] = -1
        self.reward_table[3, 1] = -1
        #self.reward_table[3, 3] = 1
        self.reward_table[4, 2] = -1
        self.start_state = np.array([4, 4])
        self.end_state = np.array([1, 1])

        self.player_state = self.start_state.copy()

    def get_reward(self, state):
        return self.reward_table[tuple(state)]

    def get_state(self, action):
        next_state = self.player_state
        if action == self.UP and next_state[0] > 0:
            next_state[0] -= 1
        if action == self.DOWN and next_state[0] < self.HEIGHT-1:
            next_state[0] += 1
        if action == self.LEFT and next_state[1] > 0:
            next_state -= 1
        if action == self.RIGHT and next_state[1] < self.WIDTH-1:
            next_state += 1
        return next_state

    def get_possible_actions(self):
        possible_actions = []
        if self.player_state[0] > 0:
            possible_actions.append(self.UP)
        if self.player_state[0] < self.HEIGHT-1:
            possible_actions.append(self.DOWN)
        if self.player_state[1] > 0:
            possible_actions.append(self.LEFT)
        if self.player_state[1] < self.WIDTH-1:
            possible_actions.append(self.RIGHT)
        return possible_actions

    def move(self, action):
        if action == self.UP:
            self.player_state[0] -= 1
        if action == self.DOWN:
            self.player_state[0] += 1
        if action == self.LEFT:
            self.player_state[1] -= 1
        if action == self.RIGHT:
            self.player_state[1] += 1
        reward = self.get_reward(self.get_player_state())
        done = np.array_equal(self.end_state, self.player_state)
        return reward, tuple(self.player_state), done

    def get_player_state(self):
        return list(self.player_state.copy())

    def reset(self):
        self.player_state = self.start_state.copy()