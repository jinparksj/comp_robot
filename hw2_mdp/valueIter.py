import random
from hw2_mdp.env3.gridworld_val import DisplayGrid, Env
import numpy as np
import time

class ValueIteration():
    def __init__(self, env, discount_factor = 0.9):
        self.env = env
        self.value_table = [[0.] * env.width for _ in range(env.height)]
        self.heading_table_forward = [[[0, 0, 0]] * env.width for _ in range(env.height)]
        self.heading_table_backward = [[[0, 0, 0]] * env.width for _ in range(env.height)]
        self.robot_head_direction = 6

        self.discount_factor = discount_factor


    def value_iteration(self):
        next_value_table = [[0.] * self.env.width for _ in range(self.env.height)]

        #Bellman Equation for every state
        for state in self.env.get_all_states():
            if state == [1, 3]: #goal location
                next_value_table[state[0]][state[1]] = 0.
                continue
            value_list = []

            for action in self.env.possible_actions: #north, east, south, west -> all checks
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + self.discount_factor * next_value))
            # return maximum value
            next_value_table[state[0]][state[1]] = round(max(value_list), 5)

        self.value_table = next_value_table

    def get_action(self, state):
        actionList = []
        max_value = -999999

        if state == [1, 3]:
            return []



        #[reward + discount factor * (next state value function)]
        for action in self.env.possible_actions:  # [0, 1, 2, 3] #north, east, south, west
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)

            if value > max_value:
                actionList.clear()
                actionList.append(action)
                max_value = value
            elif value == max_value:
                actionList.append(action)

            if actionList == [0]:
                self.heading_table_forward[state[0]][state[1]] = [11, 0, 1]
                self.heading_table_backward[state[0]][state[1]] = [5, 6, 7]
            elif actionList == [1]:
                self.heading_table_forward[state[0]][state[1]] = [2, 3, 4]
                self.heading_table_backward[state[0]][state[1]] = [8, 9, 10]
            elif actionList == [2]:
                self.heading_table_forward[state[0]][state[1]] = [5, 6, 7]
                self.heading_table_backward[state[0]][state[1]] = [11, 0, 1]
            elif actionList == [3]:
                self.heading_table_forward[state[0]][state[1]] = [8, 9, 10]
                self.heading_table_backward[state[0]][state[1]] = [2, 3, 4]
            else:
                self.heading_table_forward[state[0]][state[1]] = ['not decided', 'not decided', 'not decided']
                self.heading_table_backward[state[0]][state[1]] = ['not decided', 'not decided', 'not decided']

        return actionList


    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 5)


if __name__ == '__main__':
    env = Env()
    value_iteration = ValueIteration(env, 0.9)
    grid_world = DisplayGrid(value_iteration)
    # grid_world.mainloop()

    # time check - analysis

    start = time.time()
    temp_value_table = [[1.] * env.width for _ in range(env.height)]
    for i in range(10):
        while temp_value_table != value_iteration.value_table:
            temp_value_table = value_iteration.value_table
            grid_world.calculate_value()


        final = time.time()

        operating_time = final-start
        print(operating_time)


