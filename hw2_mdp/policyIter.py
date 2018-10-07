import random
from hw2_mdp.env2.gridworld import DisplayGrid, Env

class PolicyIteration():
    def __init__(self, env, discount_factor = 0.9):
        self.env = env
        self.value_table = [[0.] * env.width for _ in range(env.height)]
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]

        self.policy_table[1][3] = [] #location of goal
        self.discount_factor = discount_factor

    def policy_evaluation(self):
        next_value_table = [[0.] * self.env.width for _ in range(self.env.height)]

        #Bellman Equation for every state
        for state in self.env.get_all_states():
            value = 0.

            if state == [1, 3]: #goal location
                next_value_table[state[0]][state[1]] = value
                continue

            for action in self.env.possible_actions: #north, east, south, west -> all checks
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = value + (self.get_policy(state)[action] * (reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = next_value_table

    def get_policy(self, state):
        if state == [1, 3]:
            return 0.
        return self.policy_table[state[0]][state[1]]


    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

