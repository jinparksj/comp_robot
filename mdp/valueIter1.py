import random
from mdp.env5.gridworld_val import DisplayGrid, Env
import numpy as np
import time

class ValueIteration():
    def __init__(self, env, discount_factor = 0.9):
        self.env = env
        self.value_table = [[[0. for _ in range(env.heading)] for _ in  range(env.width)] for _ in range(env.height)]
        self.prob_table = [[[[[[[0. for _ in range(env.heading)] for _ in range(env.width)] for _ in range(env.height)] \
            for _ in range(len(env.possible_actions))] for _ in range(env.heading)] for _ in range(env.width)] for _ in range(env.height)]

        self.robot_head_direction = 6 #initial condition
        self.discount_factor = discount_factor
        self.pe = 0


    def transition_prob(self):

        for state in self.env.get_all_states():#state[row][ccolumn][heading]
            #state[0]: row, state[1]: col, state[2]: h

            if '[1, 3, ' in str(state): #goal location
                self.prob_table[state[0]][state[1]][state[2]] = 0.
                continue


            # move forward left, move forward right, move forward no turn, : 0, 1, 2
            # move backward left, move backward right, move backward no turn, : 3, 4, 5
            # no move no turn: 6
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                self.prob_table[state[0]][state[1]][state[2]][action][next_state[0]][next_state[1]][next_state[2]] = 1 - 2 * self.pe

                state_pe1 = [state[0], state[1], np.mod(state[2] + 1, 12)]
                next_state_pe1 = self.env.state_after_action(state_pe1, action)
                self.prob_table[state_pe1[0]][state_pe1[1]][state_pe1[2]][action] \
                    [next_state_pe1[0]][next_state_pe1[1]][next_state_pe1[2]] += self.pe

                state_pe2 = [state[0], state[1], np.mod(state[2] - 1, 12)]
                next_state_pe2 = self.env.state_after_action(state_pe2, action)
                self.prob_table[state_pe2[0]][state_pe2[1]][state_pe2[2]][action]\
                    [next_state_pe2[0]][next_state_pe2[1]][next_state_pe2[2]] += self.pe

    # def transition_prob(self):
    #
    #     noise = abs(np.random.random(1))
    #
    #     if noise < self.pe:
    #         prerot = 1  # pre-rotation right
    #     elif self.pe <= noise and noise <= 2 * self.pe:
    #         prerot = -1  # pre-rotation left
    #     else:
    #         prerot = 0
    #
    #
    #
    #     for state in self.env.get_all_states():#state[row][ccolumn][heading]
    #         #state[0]: row, state[1]: col, state[2]: h
    #
    #         if '[1, 3, ' in str(state): #goal location
    #             self.prob_table[state[0]][state[1]][state[2]] = 0.
    #             continue
    #
    #
    #         # move forward left, move forward right, move forward no turn, : 0, 1, 2
    #         # move backward left, move backward right, move backward no turn, : 3, 4, 5
    #         # no move no turn: 6
    #         for action in self.env.possible_actions:
    #             if action != 6:
    #                 state[2] = np.mod(state[2] + prerot, 12)
    #
    #             next_state = self.env.state_after_action(state, action)
    #             self.prob_table[state[0]][state[1]][state[2]][action][next_state[0]][next_state[1]][next_state[2]] = 1



    def value_iteration(self):
        next_value_table = [[[0. for _ in range(env.heading)] for _ in range(env.width)] for _ in range(env.height)]

        # Bellman Equation for every state
        for state in self.env.get_all_states():  # state[row][ccolumn][heading]
            # state[0]: row, state[1]: col, state[2]: h
            if '[1, 3, ' in str(state):  # goal location
                next_value_table[state[0]][state[1]][state[2]] = 0.
                continue
            value_list = []

            # move forward left, move forward right, move forward no turn, : 0, 1, 2
            # move backward left, move backward right, move backward no turn, : 3, 4, 5
            # no move no turn: 6

            for action in self.env.possible_actions: #north, east, south, west -> all checks
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                transition_prob = self.prob_table[state[0]][state[1]][state[2]][action][next_state[0]][next_state[1]][next_state[2]]
                value_list.append((reward + self.discount_factor * next_value) * transition_prob)
            # return maximum value
            next_value_table[state[0]][state[1]][state[2]] = round(max(value_list), 2)

        self.value_table = next_value_table

    def get_action(self, state):
        actionList = []
        max_value = -999999

        if '[1, 3, ' in str(state):
            return []

        #[reward + discount factor * (next state value function)]
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            transition_prob = self.prob_table[state[0]][state[1]][state[2]][action][next_state[0]][next_state[1]][next_state[2]]
            value = (reward + self.discount_factor * next_value) * transition_prob

            if value > max_value:
                actionList.clear()
                actionList.append(action)
                max_value = value
            elif value == max_value:
                actionList.append(action)

        return actionList


    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]][state[2]], 2)


if __name__ == '__main__':
    env = Env()
    value_iteration = ValueIteration(env, 0.9)
    grid_world = DisplayGrid(value_iteration)
    value_iteration.transition_prob()
    # grid_world.mainloop()

    # time check - analysis

    start = time.time()
    temp_value_table = [[[1. for _ in range(env.heading)] for _ in range(env.width)] for _ in range(env.height)]

    while temp_value_table != value_iteration.value_table:
        temp_value_table = value_iteration.value_table
        grid_world.calculate_value()
        # print(temp_value_table)
    grid_world.move_by_policy()

    final = time.time()

    operating_time = final-start
    print(operating_time)
    grid_world.mainloop()

