import tkinter as tk
from tkinter import Button
import time
import numpy as np
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 100
HEIGHT = 6
WIDTH = 6
HEADING = 12
TRANSITION_PROB = 1
POSSIBLE_ACTIONS = [0, 1, 2] #move forward, no move, move backward
ACTIONS = [1, 0, -1]
REWARDS = []


class DisplayGrid(tk.Tk):
    def __init__(self, agent):
        super(DisplayGrid, self).__init__()
        self.title('MDP')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT + 50)) #50 for button space
        self.texts = []
        self.arrows = []
        self.env = Env()
        self.agent = agent
        self.evalCount = 0
        self.improvCount = 0
        self.is_moving = 0
        self.pe = 0
        (self.north, self.east, self.south, self.west), self.shapes = self.load_images()
        #shapes: 0-robot, 1-border, 2-lane, 3-goal
        self.grid = self._build_grid()

        self.text_reward(1, 3, "R : 1.0")
        self.text_reward(1, 2, "R : -1.0")
        self.text_reward(2, 2, "R : -1.0")
        self.text_reward(3, 2, "R : -1.0")
        self.text_reward(1, 4, "R : -1.0")
        self.text_reward(2, 4, "R : -1.0")
        self.text_reward(3, 4, "R : -1.0")

        self.text_reward(0, 0, "R : -100.0")
        self.text_reward(1, 0, "R : -100.0")
        self.text_reward(2, 0, "R : -100.0")
        self.text_reward(3, 0, "R : -100.0")
        self.text_reward(4, 0, "R : -100.0")
        self.text_reward(5, 0, "R : -100.0")

        self.text_reward(0, 5, "R : -100.0")
        self.text_reward(1, 5, "R : -100.0")
        self.text_reward(2, 5, "R : -100.0")
        self.text_reward(3, 5, "R : -100.0")
        self.text_reward(4, 5, "R : -100.0")
        self.text_reward(5, 5, "R : -100.0")

        self.text_reward(0, 1, "R : -100.0")
        self.text_reward(0, 2, "R : -100.0")
        self.text_reward(0, 3, "R : -100.0")
        self.text_reward(0, 4, "R : -100.0")

        self.text_reward(5, 1, "R : -100.0")
        self.text_reward(5, 2, "R : -100.0")
        self.text_reward(5, 3, "R : -100.0")
        self.text_reward(5, 4, "R : -100.0")

    def load_images(self):
        north = PhotoImage(Image.open('image/north.png').resize((10, 10)))
        east = PhotoImage(Image.open('image/east.png').resize((10, 10)))
        south = PhotoImage(Image.open('image/south.png').resize((10, 10)))
        west = PhotoImage(Image.open('image/west.png').resize((10, 10)))
        robot = PhotoImage(Image.open('image/robot.png').resize((65, 65)))
        border = PhotoImage(Image.open('image/border.png').resize((65, 65)))
        lane = PhotoImage(Image.open('image/lane.png').resize((65, 65)))
        goal = PhotoImage(Image.open('image/goal.png').resize((65, 65)))

        return (north, east, south, west), (robot, border, lane, goal)

    def _build_grid(self):
        grid = tk.Canvas(self, bg = 'white', height = HEIGHT * UNIT, width = WIDTH * UNIT)

        #buttons
        #1. Policy Evaluation
        eval_button = Button(self, text = "P.Evaluate", command = self.evaluate_policy)
        eval_button.configure(width = 10, activebackground="#b5e533")
        grid.create_window(WIDTH * UNIT * 0.13, HEIGHT * UNIT + 13, window=eval_button)

        #2. Policy Improvement
        pol_button = Button(self, text = "P.Improve", command = self.improve_policy)
        pol_button.configure(width = 10, activebackground="#b5e533")
        grid.create_window(WIDTH * UNIT * 0.37, HEIGHT * UNIT + 13, window=pol_button)

        # 3. Move
        pol_button = Button(self, text="Move", command=self.move_by_policy)
        pol_button.configure(width=10, activebackground="#b5e533")
        grid.create_window(WIDTH * UNIT * 0.62, HEIGHT * UNIT + 13, window=pol_button)

        # 4. Reset
        pol_button = Button(self, text="Reset", command=self.reset)
        pol_button.configure(width=10,activebackground="#b5e533")
        grid.create_window(WIDTH * UNIT * 0.87, HEIGHT * UNIT + 13, window=pol_button)

        # Create grid
        for col in range(0, WIDTH * UNIT, UNIT): #draw vertical line
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            grid.create_line(x0, y0, x1, y1)

        for row in range(0, HEIGHT * UNIT, UNIT): #draw horizontal line
            x0, y0, x1, y1 = 0, row, HEIGHT * UNIT, row
            grid.create_line(x0, y0, x1, y1)

        # Draw images to grid
        #1. Robot
        self.robot = grid.create_image(150, 150, image = self.shapes[0])
        #robot, border, lane, goal = shapes

        #2. Borders
        grid.create_image(50, 50, image=self.shapes[1])
        grid.create_image(150, 50, image=self.shapes[1])
        grid.create_image(250, 50, image=self.shapes[1])
        grid.create_image(350, 50, image=self.shapes[1])
        grid.create_image(450, 50, image=self.shapes[1])
        grid.create_image(550, 50, image=self.shapes[1])

        grid.create_image(50, 550, image=self.shapes[1])
        grid.create_image(150, 550, image=self.shapes[1])
        grid.create_image(250, 550, image=self.shapes[1])
        grid.create_image(350, 550, image=self.shapes[1])
        grid.create_image(450, 550, image=self.shapes[1])
        grid.create_image(550, 550, image=self.shapes[1])

        grid.create_image(50, 150, image=self.shapes[1])
        grid.create_image(50, 250, image=self.shapes[1])
        grid.create_image(50, 350, image=self.shapes[1])
        grid.create_image(50, 450, image=self.shapes[1])

        grid.create_image(550, 150, image=self.shapes[1])
        grid.create_image(550, 250, image=self.shapes[1])
        grid.create_image(550, 350, image=self.shapes[1])
        grid.create_image(550, 450, image=self.shapes[1])


        #3. Lanes
        grid.create_image(250, 150, image=self.shapes[2])
        grid.create_image(250, 250, image=self.shapes[2])
        grid.create_image(250, 350, image=self.shapes[2])
        grid.create_image(450, 150, image=self.shapes[2])
        grid.create_image(450, 250, image=self.shapes[2])
        grid.create_image(450, 350, image=self.shapes[2])

        #4. Goal
        grid.create_image(350, 150, image=self.shapes[3])

        grid.pack()

        return grid

    def reset(self):
        if self.is_moving == 0:
            self.evalCount = 0
            self.improvCount = 0
            for i in self.texts:
                self.grid.delete(i)

            for i in self.arrows:
                self.grid.delete(i)

            self.agent.value_table = [[[0. for _ in range(HEADING)] for _ in  range(WIDTH)] for _ in range(HEIGHT)]
            self.agent.policy_table = [[[[0.3333, 0.3333, 0.3333] for _ in range(HEADING)] for _ in range(WIDTH)] for _ in range(HEIGHT)]
            self.agent.policy_table[1][3] = []
            x, y = self.grid.coords(self.robot)
            self.grid.move(self.robot, UNIT / 2 - x + 100, UNIT / 2 - y + 100)


    def text_value(self, row, col, contents, font = 'Times', size = 10, style = 'normal', anchor='nw'):
        origin_x, origin_y = 50, 90
        x, y = origin_x + (UNIT * col), origin_y + (UNIT * row)
        font = (font, str(size), style)
        text = self.grid.create_text(x, y, fill = 'black', text = contents, font=font, anchor=anchor)
        return self.texts.append(text)

    def text_reward(self, row, col, contents, font = 'Times', size = 10, style = 'normal', anchor = 'nw'):
        origin_x, origin_y = 5, 5
        x, y = origin_x + (UNIT * col), origin_y + (UNIT * row)
        font = (font, str(size), style)
        text = self.grid.create_text(x, y, fill = 'black', text = contents, font = font, anchor = anchor)
        return self.texts.append(text)


    def print_value_table(self, value_table):
        for i in range(HEIGHT): #row
            for j in range(WIDTH): #col
                    self.text_value(i, j, max(value_table[i][j]))


    def turn_action(self, heading, turn):
        if turn == 'left':
            heading = np.mod(heading - 1, 12)
        elif turn == 'right':
            heading = np.mod(heading + 1, 12)
        elif turn == 'noturn':
            heading = heading
        return heading


    def robot_move_display(self, action):

        noise = abs(np.random.random(1))

        if noise < self.pe:
            prerot = 1 #pre-rotation right
        elif self.pe <= noise and noise <= 2 * self.pe:
            prerot = -1 #pre-rotation left
        else:
            prerot = 0

        base_action = np.array([0, 0])
        location = self.find_robot() # robot's y and x position
        self.render()
        # move forward, no move, backward: 1, 0, -1


        self.agent.robot_head_direction = self.agent.robot_head_direction + prerot

        if

        if self.agent.robot_head_direction in self.agent.heading_table_backward[location[0]][location[1]]: #forward table
            base_action = self.move_turn(location, action, base_action)

        elif self.agent.robot_head_direction in self.agent.heading_table_backward[location[0]][location[1]]: #backward table
            base_action = self.move_turn(location, action, base_action)

        else: #no move and turn
            self.agent.robot_head_direction = self.agent.robot_head_direction - prerot #no prerotation
            if abs(self.agent.robot_head_direction - self.agent.heading_table_forward[location[0]][location[1]][1]) >= \
                abs(self.agent.robot_head_direction - self.agent.heading_table_backward[location[0]][location[1]][1]): #backward
                while self.agent.robot_head_direction not in self.agent.heading_table_backward[location[0]][location[1]]:
                    self.no_move_left_turn(base_action)
                    print('no move left turn', self.agent.robot_head_direction)

                base_action = self.move_turn(location, action, base_action)



            elif abs(self.agent.robot_head_direction - self.agent.heading_table_forward[location[0]][location[1]][1]) < \
                abs(self.agent.robot_head_direction - self.agent.heading_table_backward[location[0]][location[1]][1]):
                while self.agent.robot_head_direction not in self.agent.heading_table_forward[location[0]][location[1]]:
                    self.no_move_right_turn(base_action)
                    print('no move right turn', self.agent.robot_head_direction)

                base_action = self.move_turn(location, action, base_action)

        self.grid.move(self.robot, base_action[1], base_action[0]) #move(tagOrId, xAmount, yAmount)
        print('move',self.agent.robot_head_direction)


    def move_turn(self, location, action, base_action):
        if action == 0 and location[0] > 0:  # north_y
            base_action[0] -= UNIT
            if self.agent.heading_table_forward[location[0] - 1][location[1]] == [8, 9, 10] \
                    and location[0] - 1 > 0:  # left
                self.turn_action(self.agent.robot_head_direction, 'left')
                print('lookahead turning')
            if self.agent.heading_table_forward[location[0] - 1][location[1]] == [2, 3, 4] \
                    and location[0] - 1 > 0:  # right
                self.turn_action(self.agent.robot_head_direction, 'right')
                print('lookahead turning')

        elif action == 1 and location[1] < WIDTH - 1:  # east_x
            base_action[1] += UNIT
            if self.agent.heading_table_forward[location[0]][location[1] + 1] == [11, 0, 1] \
                    and location[1] + 1 < WIDTH - 1:  # left
                self.turn_action(self.agent.robot_head_direction, 'left')
                print('lookahead turning')
            if self.agent.heading_table_forward[location[0]][location[1] + 1] == [2, 3, 4] \
                    and location[1] + 1 < WIDTH - 1:  # right
                self.turn_action(self.agent.robot_head_direction, 'right')
                print('lookahead turning')

        elif action == 2 and location[0] < HEIGHT - 1:  # south_y
            base_action[0] += UNIT
            if self.agent.heading_table_forward[location[0] + 1][location[1]] == [2, 3, 4] \
                    and location[0] + 1 < HEIGHT - 1:  # left
                self.turn_action(self.agent.robot_head_direction, 'left')
                print('lookahead turning')
            if self.agent.heading_table_forward[location[0] + 1][location[1]] == [8, 9, 10] \
                    and location[0] + 1 < HEIGHT - 1:  # right
                self.turn_action(self.agent.robot_head_direction, 'right')
                print('lookahead turning')

        elif action == 3 and location[1] > 0:  # west_x
            base_action[1] -= UNIT
            if self.agent.heading_table_forward[location[0]][location[1] + 1] == [5, 6, 7] \
                    and location[1] - 1 > 0:  # left
                self.turn_action(self.agent.robot_head_direction, 'left')
                print('lookahead turning')

            if self.agent.heading_table_forward[location[0]][location[1] + 1] == [11, 0, 1] \
                    and location[1] - 1 > 0:  # right
                self.turn_action(self.agent.robot_head_direction, 'right')
                print('lookahead turning')

        return base_action

    def no_move_left_turn(self, base_action):
        self.agent.robot_head_direction = self.turn_action(self.agent.robot_head_direction, 'left')
        base_action = np.array([0, 0])
        self.grid.move(self.robot, base_action[1], base_action[0])  # move(tagOrId, xAmount, yAmount)

    def no_move_right_turn(self, base_action):
        self.agent.robot_head_direction = self.turn_action(self.agent.robot_head_direction, 'right')
        base_action = np.array([0, 0])
        self.grid.move(self.robot, base_action[1], base_action[0])  # move(tagOrId, xAmount, yAmount)


    def find_robot(self):
        temp = self.grid.coords(self.robot)
        x = (temp[0] / 100) - 0.5
        y = (temp[1] / 100) - 0.5
        return int(y), int(x)

    def move_by_policy(self):
        if self.improvCount != 0 and self.is_moving != 1:
            self.is_moving = 1

            x, y = self.grid.coords(self.robot)
            self.grid.move(self.robot, UNIT / 2 - x + 100, UNIT / 2 - y + 100)

            y, x = self.find_robot()
            while len(self.agent.policy_table[y][x]) != 0:
                self.after(100, self.robot_move_display(self.agent.get_action([y, x])))
                y, x = self.find_robot()
            self.is_moving = 0

    def evaluate_policy(self):
        self.evalCount += 1
        for i in self.texts:
            self.grid.delete(i)
        self.agent.policy_evaluation()
        # self.print_value_table(self.agent.value_table)


    def improve_policy(self):
        self.improvCount += 1
        for i in self.arrows:
            self.grid.delete(i)
        self.agent.policy_improvement()
        # self.draw_from_policy(self.agent.policy_table)

    def draw_from_policy(self, policy_table):
        for i in range(HEIGHT):
            for j in range(WIDTH):
                self.draw_one_arrow(i, j, policy_table[i][j])

    def draw_one_arrow(self, row, col, policy):
        if row == 1 and col == 3:
            return

        if policy[0] > 0: #north
            origin_x, origin_y = 50 + (UNIT * col), 10 + (UNIT * row)
            self.arrows.append(self.grid.create_image(origin_x, origin_y, image = self.north))

        if policy[1] > 0: #east
            origin_x, origin_y = 90 + (UNIT * col), 50 + (UNIT * row)
            self.arrows.append(self.grid.create_image(origin_x, origin_y, image = self.east))

        if policy[2] > 0: #south
            origin_x, origin_y = 50 + (UNIT * col), 90 + (UNIT * row)
            self.arrows.append(self.grid.create_image(origin_x, origin_y, image = self.south))

        if policy[3] > 0: #west
            origin_x, origin_y = 10 + (UNIT * col), 50 + (UNIT * row)
            self.arrows.append(self.grid.create_image(origin_x, origin_y, image = self.west))


    def render(self):
        time.sleep(0.5)
        self.grid.tag_raise(self.robot)
        self.update()


class Env():
    def __init__(self):
        self.transition_prob = TRANSITION_PROB
        self.width = WIDTH
        self.height = HEIGHT
        self.heading = HEADING

        self.reward = [[[[0] for _ in range(HEADING)] for _ in range(WIDTH)] for _ in range(HEIGHT)]
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward[1][3][:] = [[1]] * 12 #reward 1 for goal

        self.reward[1][2][:] = [[-1]] * 12  # reward -1 for lanes
        self.reward[2][2][:] = [[-1]] * 12
        self.reward[3][2][:] = [[-1]] * 12
        self.reward[1][4][:] = [[-1]] * 12
        self.reward[2][4][:] = [[-1]] * 12
        self.reward[3][4][:] = [[-1]] * 12

        self.reward[0][0][:] = [[-100]] * 12  # reward [[-100]] * 12 for borders
        self.reward[0][1][:] = [[-100]] * 12
        self.reward[0][2][:] = [[-100]] * 12
        self.reward[0][3][:] = [[-100]] * 12
        self.reward[0][4][:] = [[-100]] * 12
        self.reward[0][5][:] = [[-100]] * 12

        self.reward[5][0][:] = [[-100]] * 12
        self.reward[5][1][:] = [[-100]] * 12
        self.reward[5][2][:] = [[-100]] * 12
        self.reward[5][3][:] = [[-100]] * 12
        self.reward[5][4][:] = [[-100]] * 12
        self.reward[5][5][:] = [[-100]] * 12

        self.reward[1][0][:] = [[-100]] * 12
        self.reward[2][0][:] = [[-100]] * 12
        self.reward[3][0][:] = [[-100]] * 12
        self.reward[4][0][:] = [[-100]] * 12

        self.reward[1][5][:] = [[-100]] * 12
        self.reward[2][5][:] = [[-100]] * 12
        self.reward[3][5][:] = [[-100]] * 12
        self.reward[4][5][:] = [[-100]] * 12


        self.all_state = []



        for x in range(WIDTH):
            for y in range(HEIGHT):
                for h in range(HEADING):
                    state = [y, x, h] #y is row, x is column
                    self.all_state.append(state)


    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        reward = self.reward[next_state[0]][next_state[1]][next_state[2]][0]
        return reward

    def state_after_action(self, state, action_index): #state - [r, c, h]
        #move forward: 1, no move: 0, move backward: -1
        if state[2] in [11, 0, 1]: #north_y
            if ACTIONS[action_index] == 1: #action index: 0, 1, 2, #ACTIONS: 1, 0, -1
                action = (-1, 0)
            elif ACTIONS[action_index] == 0:
                action = (0, 0)
            elif ACTIONS[action_index] == -1:
                action = (1, 0)
        elif state[2] in [2, 3, 4]:
            if ACTIONS[action_index] == 1: # forward, action index: 0, 1, 2, #ACTIONS: 1, 0, -1
                action = (0, 1)
            elif ACTIONS[action_index] == 0: #no move
                action = (0, 0)
            elif ACTIONS[action_index] == -1: #backward
                action = (0, -1)
        elif state[2] in [5, 6, 7]:
            if ACTIONS[action_index] == 1: #action index: 0, 1, 2, #ACTIONS: 1, 0, -1
                action = (1, 0)
            elif ACTIONS[action_index] == 0: #no move
                action = (0, 0)
            elif ACTIONS[action_index] == -1: #backward
                action = (-1, 0)
        elif state[2] in [8, 9, 10]:
            if ACTIONS[action_index] == 1: # forward, action index: 0, 1, 2, #ACTIONS: 1, 0, -1
                action = (0, -1)
            elif ACTIONS[action_index] == 0: #no move
                action = (0, 0)
            elif ACTIONS[action_index] == -1: #backward
                action = (0, 1)
        return self.check_boundary([state[0] + action[0], state[1] + action[1], state[2]])

    @staticmethod
    def check_boundary(state):
        if state[0] < 0:
            state[0] = 0
        elif state[0] > HEIGHT - 1:
            state[0] = HEIGHT - 1
        else:
            state[0] = state[0]

        if state[1] < 0:
            state[1] = 0
        elif state[1] > WIDTH - 1:
            state[1] = WIDTH-1
        else:
            state[1] = state[1]
        return state

    def get_all_states(self):
        return self.all_state #state[row][ccolumn][heading]