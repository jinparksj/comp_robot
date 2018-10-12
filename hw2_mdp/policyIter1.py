import random
from hw2_mdp.env4.gridworld import DisplayGrid, Env
import numpy as np
import time

class PolicyIteration():
    def __init__(self, env, discount_factor = 0.9):
        self.env = env
        self.value_table = [[[0. for _ in range(env.heading)] for _ in  range(env.width)] for _ in range(env.height)]
        self.policy_table = [[[[0.3333, 0.3333, 0.3333] for _ in range(env.heading)] for _ in range(env.width)] for _ in range(env.height)]

        self.robot_head_direction = 6 #initial condition
        self.policy_table[1][3][:] = []  #location of goal
        self.discount_factor = discount_factor


    def policy_evaluation(self):
        next_value_table = [[[0. for _ in range(env.heading)] for _ in  range(env.width)] for _ in range(env.height)]

        #Bellman Equation for every state
        for state in self.env.get_all_states():#state[row][ccolumn][heading]
            value = 0.
            #state[0]: row, state[1]: col, state[2]: h
            if '[1, 3, ' in str(state): #goal location
                next_value_table[state[0]][state[1]][state[2]] = value
                continue

            for action in self.env.possible_actions: #move forward, no move, move backward -> all checks!, possible_actions:0, 1, 2 index
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = value + (self.get_policy(state)[action] * (reward + self.discount_factor * next_value))
            next_value_table[state[0]][state[1]][state[2]] = round(value, 4)

        self.value_table = next_value_table

    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if '[1, 3' in str(state):
                continue
            value = -999999
            max_index = []
            result = [0.0, 0.0, 0.0] #initialize policy

            #[reward + discount factor * (next state value function)]
            for index, action in enumerate(self.env.possible_actions):  # [0, 1, 2] action index of move forward, no move, backward
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

            prob = round(1 / len(max_index), 3)

            for index in max_index:
                result[index] = prob

            next_policy[state[0]][state[1]][state[2]] = result

        self.policy_table = next_policy

    def get_policy(self, state):
        if '[1, 3' in str(state):
            return 0.
        return self.policy_table[state[0]][state[1]][state[2]]


    #get action index 0, 1, 2 from policy table
    def get_action(self, state):
        random_pick = random.randrange(100) / 100
        policy = self.get_policy(state) #[1, 0, 0]: move forward, [0, 1, 0]: no move, [0, 0, 1]: move backward
        policy_sum = 0.0
        # return the action in the index
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]][state[2]], 4)


if __name__ == '__main__':
    env = Env()
    policy_iteration = PolicyIteration(env, 0.7)
    grid_world = DisplayGrid(policy_iteration)
    # grid_world.mainloop()

    #time check - analysis


    temp_policy_table = []
    temp_value_table = [[[0. for _ in range(env.heading)] for _ in  range(env.width)] for _ in range(env.height)]
    for i in range(10):
        start = time.time()
        while temp_policy_table != policy_iteration.policy_table or temp_value_table != policy_iteration.value_table:
            temp_policy_table = policy_iteration.policy_table
            temp_value_table = policy_iteration.value_table
            grid_world.evaluate_policy()
            grid_world.improve_policy()


        # print(policy_iteration.value_table)
        final = time.time()

        operating_time = final-start
        print(operating_time)

    print('done')