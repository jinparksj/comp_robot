import tkinter as tk
from tkinter import Button
import time
import numpy as np
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 100
HEIGHT = 6
WIDTH = 6
TRANSITION_PROB = 1
POSSIBLE_ACTIONS = [0, 1, 2, 3] #north, east, south, west
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
REWARDS = []

help(tk.Tk.geometry)

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
        (self.north, self.east, self.south, self.right), self.shapes = self.load_images()
        #shapes: 0-robot, 1-border, 2-lane, 3-goal
        self.grid = self._build_grid()

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
        grid.create_window(WIDTH * UNIT * 0.37, HEIGHT * UNIT + 13, window=eval_button)




    def text_value

    def print_value_table(self, value_table):
        for i in range(WIDTH):
            for j in range(HEIGHT):
                self.text_value(i, )

    def evaluate_policy(self):
        self.evalCount += 1
        for i in self.texts:
            self.grid.delete(i)
        self.agent.policy_evaluation()
        self.print_value_table(self.agent.value_table)


    def improve_policy(self):
        self.improvCount += 1
        for i in self.arrows:
            self.grid.delete(i)
        self.agent.policy_improvement()
        self.draw_from_policy(self.agent.policy_table)

    def load_images(self):
        north = PhotoImage(Image.open('../image/north.png').resize((10, 10)))
        east = PhotoImage(Image.open('../image/east.png').resize((10, 10)))
        south = PhotoImage(Image.open('../image/south.png').resize((10, 10)))
        west = PhotoImage(Image.open('../image/west.png').resize((10, 10)))
        robot = PhotoImage(Image.open('../image/robot.png')).resize((65, 65))
        border = PhotoImage(Image.open('../image/border.png')).resize((65, 65))
        lane = PhotoImage(Image.open('../image/lane.png')).resize((65, 65))
        goal = PhotoImage(Image.open('../image/goal.png')).resize((65, 65))
        return (north, east, south, west), (robot, border, lane, goal)

class Env():
    def __init__(self):
        self.transition_prob = TRANSITION_PROB
        self.width = WIDTH
        self.height = HEIGHT
        self.reward = [[0] * WIDTH for _ in range(HEIGHT)]
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward[1][3] = 1 #reward 1 for goal 

        self.reward[1][2] = -1  # reward -1 for lanes
        self.reward[2][2] = -1  
        self.reward[3][2] = -1  
        self.reward[1][4] = -1  
        self.reward[2][4] = -1  
        self.reward[3][4] = -1  

        self.reward[0][0] = -100  # reward -100 for borders
        self.reward[0][1] = -100  
        self.reward[0][2] = -100  
        self.reward[0][3] = -100  
        self.reward[0][4] = -100  
        self.reward[0][5] = -100  

        self.reward[5][0] = -100  
        self.reward[5][1] = -100  
        self.reward[5][2] = -100  
        self.reward[5][3] = -100  
        self.reward[5][4] = -100  
        self.reward[5][5] = -100  

        self.reward[1][0] = -100  
        self.reward[2][0] = -100  
        self.reward[3][0] = -100  
        self.reward[4][0] = -100  

        self.reward[1][5] = -100  
        self.reward[2][5] = -100  
        self.reward[3][5] = -100  
        self.reward[4][5] = -100  


        self.all_state = []

        for x in range(WIDTH):
            for y in range(HEIGHT):
                state = [y, x] #y is row, x is column
                self.all_state.append(state)


    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action_index):
        action = ACTIONS[action_index] #north, east, south, west
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

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
        return self.all_state