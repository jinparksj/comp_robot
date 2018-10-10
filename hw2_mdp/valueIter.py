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
#================================================================
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [1, 3]:
                continue
            value = -99999
            max_index = []
            result = [0.0, 0.0, 0.0, 0.0] #initialize policy

            #[reward + discount factor * (next state value function)]
            for index, action in enumerate(self.env.possible_actions):  # [0, 1, 2, 3] #north, east, south, west
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp = reward + self.discount_factor * next_value

                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            prob = 1 / len(max_index)

            for index in max_index:
                result[index] = prob

            next_policy[state[0]][state[1]] = result

            if next_policy[state[0]][state[1]] == [1., 0., 0., 0.]:
                self.heading_table_forward[state[0]][state[1]] = [11, 0, 1]
                self.heading_table_backward[state[0]][state[1]] = [5, 6, 7]
            elif self.policy_table[state[0]][state[1]] == [0., 1., 0., 0.]:
                self.heading_table_forward[state[0]][state[1]] = [2, 3, 4]
                self.heading_table_backward[state[0]][state[1]] = [8, 9, 10]
            elif self.policy_table[state[0]][state[1]] == [0., 0., 1., 0.]:
                self.heading_table_forward[state[0]][state[1]] = [5, 6, 7]
                self.heading_table_backward[state[0]][state[1]] = [11, 0, 1]
            elif self.policy_table[state[0]][state[1]] == [0., 0., 0., 1.]:
                self.heading_table_forward[state[0]][state[1]] = [8, 9, 10]
                self.heading_table_backward[state[0]][state[1]] = [2, 3, 4]
            else:
                self.heading_table_forward[state[0]][state[1]] = ['not decided', 'not decided', 'not decided']
                self.heading_table_backward[state[0]][state[1]] = ['not decided', 'not decided', 'not decided']

        # print(self.heading_table_forward)
        # print(self.heading_table_backward)
        self.policy_table = next_policy

    def get_policy(self, state):
        if state == [1, 3]:
            return 0.
        return self.policy_table[state[0]][state[1]]



    def get_action(self, state):
        random_pick = random.randrange(100) / 100
        policy = self.get_policy(state)
        policy_sum = 0.0
        # return the action in the index
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    def pol2heading(self, state):
        for i in range(self.env.height):
            for j in range(self.env.width):
                if self.policy_table[i][j] == [1., 0., 0., 0.]:
                    self.heading_table[i][j] = [11, 0, 1]
                elif self.policy_table[i][j] == [0., 1., 0., 0.]:
                    self.heading_table[i][j] = [2, 3, 4]
                elif self.policy_table[i][j] == [0., 0., 1., 0.]:
                    self.heading_table[i][j] = [5, 6, 7]
                elif self.policy_table[i][j] == [0., 0., 0., 1.]:
                    self.heading_table[i][j] = [8, 9, 10]


    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 5)


if __name__ == '__main__':
    env = Env()
    policy_iteration = PolicyIteration(env, 0.9)
    grid_world = DisplayGrid(policy_iteration)
    grid_world.mainloop()