import random
from mdp.env4.gridworld import DisplayGrid, Env
import numpy as np
import time

class PolicyIteration():
    def __init__(self, env, discount_factor = 0.9):
        self.env = env
        self.value_table = [[[0. for _ in range(env.heading)] for _ in  range(env.width)] for _ in range(env.height)]
        self.policy_table = [[[[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7] for _ in range(env.heading)] for _ in range(env.width)] for _ in range(env.height)]
        self.policy_table[1][1][6] = [1., 0., 0., 0., 0., 0., 0.]
        self.policy_table[2][1][5] = [0., 0., 0., 1., 0., 0., 0.]
        self.policy_table[1][1][4] = [1., 0., 0., 0., 0., 0., 0.]
        self.policy_table[1][2][3] = [0., 0., 1., 0., 0., 0., 0.]


        self.prob_table = [[[[[[[0. for _ in range(env.heading)] for _ in range(env.width)] for _ in range(env.height)] \
            for _ in range(len(env.possible_actions))] for _ in range(env.heading)] for _ in range(env.width)] for _ in range(env.height)]


        self.robot_head_direction = 6 #initial condition
        self.policy_table[1][3][:] =[[]]  #location of goal
        self.discount_factor = discount_factor
        self.pe = 0.25

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



    def policy_evaluation(self):
        next_value_table = [[[0. for _ in range(env.heading)] for _ in  range(env.width)] for _ in range(env.height)]

        #Bellman Equation for every state
        for state in self.env.get_all_states():#state[row][ccolumn][heading]
            value = 0.
            #state[0]: row, state[1]: col, state[2]: h
            if '[1, 3, ' in str(state): #goal location
                next_value_table[state[0]][state[1]][state[2]] = value
                continue

            # move forward left, move forward right, move forward no turn, : 0, 1, 2
            # move backward left, move backward right, move backward no turn, : 3, 4, 5
            # no move no turn: 6

            for action in self.env.possible_actions: #move forward, no move, move backward -> all checks!, possible_actions:0, 1, 2 index
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                transition_prob = self.prob_table[state[0]][state[1]][state[2]][action][next_state[0]][next_state[1]][next_state[2]]
                p_test = self.get_policy(state)[action]
                value = value + (reward + self.discount_factor * next_value) * transition_prob * self.get_policy(state)[action]

            next_value_table[state[0]][state[1]][state[2]] = round(value, 2)

        self.value_table = next_value_table

    def policy_improvement(self):
        next_policy = self.policy_table

        for state in self.env.get_all_states():
            if '[1, 3' in str(state):
                continue
            value = -999
            max_index = []
            result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #initialize policy

            # move forward left, move forward right, move forward no turn, : 0, 1, 2
            # move backward left, move backward right, move backward no turn, : 3, 4, 5
            # no move no turn: 6
            temp = 0
            #[reward + discount factor * (next state value function)]
            for index, action in enumerate(self.env.possible_actions):  # [0, 1, 2] action index of move forward, no move, backward
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                transition_prob = self.prob_table[state[0]][state[1]][state[2]][action][next_state[0]][next_state[1]][next_state[2]]
                temp = (reward + self.discount_factor * next_value) * transition_prob

                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            prob = 1 / len(max_index)

            for index in max_index:
                result[index] = prob

            next_policy[state[0]][state[1]][state[2]] = result

        self.policy_table = next_policy

    def get_policy(self, state):
        if '[1, 3' in str(state):
            return 0.
        return self.policy_table[state[0]][state[1]][state[2]]

    def turn_action(self, heading, turn):
        if turn == 'left':
            heading = np.mod(heading - 1, 12)
        elif turn == 'right':
            heading = np.mod(heading + 1, 12)
        elif turn == 'noturn':
            heading = heading
        return heading


    #get action index 0, 1, 2 from policy table
    def get_action(self, state):
        policy = self.get_policy(state)
        if '[1, 3 ' in str(state):
            return

        val = 0.
        for i in range(len(policy)):
            if policy[i] > 0:
                val = val + policy[i]
                if round(val, 1) == 1:
                    return i

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]][state[2]], 2)


if __name__ == '__main__':
    env = Env()
    policy_iteration = PolicyIteration(env, 0.9)
    grid_world = DisplayGrid(policy_iteration)
    policy_iteration.transition_prob()
    # for i in range(200):
    #     grid_world.evaluate_policy()
    #     grid_world.improve_policy()

    # grid_world.mainloop()

    ###########################time check - analysis###############################


    temp_policy_table = []
    temp_value_table = [[[0. for _ in range(env.heading)] for _ in  range(env.width)] for _ in range(env.height)]

    start = time.time()
    while temp_policy_table != policy_iteration.policy_table or temp_value_table != policy_iteration.value_table:
        temp_policy_table = policy_iteration.policy_table
        temp_value_table = policy_iteration.value_table
        grid_world.evaluate_policy()
        grid_world.improve_policy()
        print(temp_value_table)

    # print(policy_iteration.value_table)
    final = time.time()

    operating_time = final-start
    print(operating_time)
    grid_world.mainloop()
    print('done')