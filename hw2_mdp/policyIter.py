import random
from hw2_mdp.env2.gridworld import DisplayGrid, Env
import numpy as np

class PolicyIteration():
    def __init__(self, env, discount_factor = 0.9):
        self.env = env
        self.value_table = [[0.] * env.width for _ in range(env.height)]
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
        self.heading_table_forward = [[[0, 0, 0]] * env.width for _ in range(env.height)]
        self.heading_table_backward = [[[0, 0, 0]] * env.width for _ in range(env.height)]
        self.robot_head_direction = 6
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

        print(self.heading_table_forward)
        print(self.heading_table_backward)
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


    # def get_action(self, state, turn, moving_direction = 'forward', pe = 0.25, heading = 6):
    #     # state - location of robot (1, 1)
    #     # 2, 3, 4: forwards as +x, 11, 0, 1: backwards: -y, 5, 6, 7: forward as +y, 8, 9, 10: backward as -x
    #     # moving_direction: 'forward', 'backward', 'nomove'
    #     # 1. moving or not moving
    #     # 2. moving "forwards" and "backwards" with direction
    #     #   - when moving, cause pre-rotation error
    #     #   - rounded to the nearest cardinal direction
    #     # 3. after moving, choose 1) turn left, 2) not turn, 3) turn right
    #     #     1) left - decrease the heading by 1
    #     #     3) right - increase the heading by 1
    #     #     2) robot can also keep the heading constant
    #     # 4. error probability pe
    #     #   if the robot chooses to move, it will first rotate by +1 or -1 with pe, before it moves
    #     #   It will not pre-rotate with 1-2*pe
    #     #   when choosing not moving, no error rotation
    #
    #     robotY = state[0]
    #     robotX = state[1]
    #
    #     noise = np.random.randn(1)
    #
    #     threshold = np.random.randn(1)
    #     policy = self.get_policy(state)
    #
    #     if noise < pe:
    #         prerot = 1  # pre-rotation right
    #     elif pe <= noise and noise <= 2 * pe:
    #         prerot = -1  # pre-rotation left
    #     else:
    #         prerot = 0
    #     # action 0, 1, 2, 3 : north, east, south, west
    #     if moving_direction == 'forward':  # forwards or backwards with pre-rotation error
    #         heading = heading + prerot
    #         if heading == 2 and robotX <= self.env.width - 2:
    #             self.turn_action(heading, turn)
    #             return 1
    #         elif heading == 3 and robotX <= self.env.width - 2:
    #             self.turn_action(heading, turn)
    #         elif heading == 4 and robot.x <= self.sizeX - 2:
    #             robot.x = np.mod(robot.x + 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 8 and robot.x >= 1:
    #             robot.x = np.mod(robot.x - 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 9 and robot.x >= 1:
    #             robot.x = np.mod(robot.x - 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 10 and robot.x >= 1:
    #             robot.x = np.mod(robot.x - 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 5 and robot.y <= self.sizeY - 2:
    #             robot.x = np.mod(robot.y + 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 6 and robot.y <= self.sizeY - 2:
    #             robot.x = np.mod(robot.y + 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 7 and robot.y <= self.sizeY - 2:
    #             robot.x = np.mod(robot.y + 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 11 and robot.y >= 1:
    #             robot.x = np.mod(robot.y - 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 0 and robot.y >= 1:
    #             robot.x = np.mod(robot.y - 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 1 and robot.y >= 1:
    #             robot.x = np.mod(robot.y - 1, 12)
    #             self.turn_action(heading, turn)
    #         else:
    #             self.turn_action(heading, turn)
    #
    #     elif moving_direction == 'backward':  # forwards or backwards with pre-rotation error
    #         heading = heading + prerot
    #
    #         if heading == 8 and robot.x <= self.sizeX - 2:
    #             robot.x = np.mod(robot.x + 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 9 and robot.x <= self.sizeX - 2:
    #             robot.x = np.mod(robot.x + 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 10 and robot.x <= self.sizeX - 2:
    #             robot.x = np.mod(robot.x + 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 2 and robot.x >= 1:
    #             robot.x = np.mod(robot.x - 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 3 and robot.x >= 1:
    #             robot.x = np.mod(robot.x - 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 4 and robot.x >= 1:
    #             robot.x = np.mod(robot.x - 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 11 and robot.y <= self.sizeY - 2:
    #             robot.x = np.mod(robot.y + 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 0 and robot.y <= self.sizeY - 2:
    #             robot.x = np.mod(robot.y + 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 1 and robot.y <= self.sizeY - 2:
    #             robot.x = np.mod(robot.y + 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 5 and robot.y >= 1:
    #             robot.x = np.mod(robot.y - 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 6 and robot.y >= 1:
    #             robot.x = np.mod(robot.y - 1, 12)
    #             self.turn_action(heading, turn)
    #         elif heading == 7 and robot.y >= 1:
    #             robot.x = np.mod(robot.y - 1, 12)
    #             self.turn_action(heading, turn)
    #         else:
    #             self.turn_action(heading, turn)
    #
    #
    #     else:  # not moving
    #         heading = heading
    #         robot.x = robot.x
    #         robot.y = robot.y
    #
    #     # need to decide action 0, 1, 2, 3 north, east, south, west



    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)


if __name__ == '__main__':
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = DisplayGrid(policy_iteration)
    grid_world.mainloop()