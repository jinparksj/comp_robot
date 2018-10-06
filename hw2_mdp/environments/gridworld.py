import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt

class gridObs():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

class gridEnv():
    def __init__(self, size = 6):
        self.sizeX = size
        self.sizeY = size
        self.actions = 12
        self.objects = []
        grid = self.reset()
        plt.imshow(grid, interpolation="nearest")

    def reset(self):
        self.objects = []

        #1. robot
        robot = gridObs(self.position('robot'), 1, 1, 2, None, 'robot')
        self.objects.append(robot)
        '''
        borderList = ['border1', 'border2', 'border3', 'border4', 'border5', 'border6', 'border7', 'border8', \
                      'border9', 'border10', 'border11', 'border12', 'border13', 'border14', 'border15', 'border16', \
                      'border17', 'border18', 'border19', 'border20']
        '''

        #2. borders
        topborderList = []
        leftborderList = []
        rightborderList = []
        bottomborderList = []

        for i in range(6):
            topborderList.append(gridObs((0, i), 1, 1, 0, -100, 'topborder{0}'.format(i)))
            self.objects.append(topborderList[i])

        for i in range(6):
            bottomborderList.append(gridObs((5, i), 1, 1, 0, -100, 'bottomborder{0}'.format(i)))
            self.objects.append(bottomborderList[i])

        for i in range(4):
            leftborderList.append(gridObs((i+1, 0), 1, 1, 0, -100, 'leftborder{0}'.format(i)))
            self.objects.append(leftborderList[i])

        for i in range(4):
            rightborderList.append(
                gridObs((i+1, 5), 1, 1, 3, -100, 'rightborder{0}'.format(i)))
            self.objects.append(rightborderList[i])

        #3. lane markers
        laneList = []
        for i in range(6):
            laneList.append(gridObs(self.position('lane{0}'.format(i)), 1, 1, 1, -1, 'lane{0}'.format(i)))
            self.objects.append(laneList[i])

        #4. goal
        goal = gridObs(self.position('goal'), 1, 1, 1, 1, 'goal')
        self.objects.append(goal)

        state = self.renderEnv()
        return state

    def position(self, name):
        iters = [ range(self.sizeX), range(self.sizeY) ]
        points = []
        for t in itertools.product(*iters):
            points.append(t)
        curPosition = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in curPosition:
                curPosition.append((objectA.x, objectA.y))
        for pos in curPosition:
            points.remove(pos)

        # for i in borderList:
        #     if 'border' in i:
        #         print(i)

        if name == 'robot':
            loc = 7
        elif 'lane' in name:
            if '0' in name:
                loc = 8
            elif '1' in name:
                loc = 10
            elif '2' in name:
                loc = 14
            elif '3' in name:
                loc = 16
            elif '4' in name:
                loc = 20
            elif '5' in name:
                loc = 22
        elif name == 'goal':
            loc = 9
        return points[loc]

    def renderEnv(self):
        grid = np.ones([self.sizeY + 2, self.sizeX + 2, 3])
        grid[1:-1, 1:-1, : ] = 0

        robot = None

        for item in self.objects:
            grid[item.y+1 : item.y + item.size + 1, item.x + 1 : item.x + item.size + 1, item.channel] = item.intensity

        if item.name == 'robot':
            robot = item

        b = scipy.misc.imresize(grid[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(grid[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(grid[:, :, 2], [84, 84, 1], interp='nearest')
        grid = np.stack([b, c, d], axis=2)

        return grid

    def turnChar(self, heading, turn): #left, right, noturn
        if turn == 'left':
            heading = np.mod(heading - 1, 12)
        elif turn == 'right':
            heading = np.mod(heading + 1, 12)
        elif turn == 'noturn':
            heading = heading
        return heading

    def moveChar(self, heading, moving, turn):
        # 2, 3, 4: forwards as +x, 11, 0, 1: backwards: -y, 5, 6, 7: forward as +y, 8, 9, 10: backward as -x
        #
        # 1. moving or not moving
        # 2. moving "forwards" and "backwards"
        #   - when moving, cause pre-rotation error
        #   - rounded to the nearest cardinal direction
        # 3. after moving, choose 1) turn left, 2) not turn, 3) turn right
        #     1) left - decrease the heading by 1
        #     3) right - increase the heading by 1
        #     2) robot can also keep the heading constant
        # 4. error probability pe
        #   if the robot chooses to move, it will first rotate by +1 or -1 with pe, before it moves
        #   It will not pre-rotate with 1-2*pe
        #   when choosing not moving, no error rotation
        #

        robot = self.objects[0]
        robotX = robot.x
        robotY = robot.y
        penalize = 0.


        if moving == 'move': #forwards or backwards with pre-rotation error
            if heading == 2 and robot.x <= self.sizeX - 2:
                robot.x = np.mod(robot.x + 1, 12)
                self.turnChar(heading, turn)
            elif heading == 3 and robot.x <= self.sizeX - 2:
                robot.x = np.mod(robot.x + 1, 12)
                self.turnChar(heading, turn)
            elif heading == 4 and robot.x <= self.sizeX - 2:
                robot.x = np.mod(robot.x + 1, 12)
                self.turnChar(heading, turn)
            elif heading == 8 and robot.x >= 1 :
                robot.x = np.mod(robot.x - 1, 12)
                self.turnChar(heading, turn)
            elif heading == 9 and robot.x >= 1:
                robot.x = np.mod(robot.x - 1, 12)
                self.turnChar(heading, turn)
            elif heading == 10 and robot.x >= 1:
                robot.x = np.mod(robot.x - 1, 12)
                self.turnChar(heading, turn)
            elif heading == 5 and robot.y <= self.sizeY - 2:
                robot.x = np.mod(robot.y + 1, 12)
                self.turnChar(heading, turn)
            elif heading == 6 and robot.y <= self.sizeY - 2:
                robot.x = np.mod(robot.y + 1, 12)
                self.turnChar(heading, turn)
            elif heading == 7 and robot.y <= self.sizeY - 2:
                robot.x = np.mod(robot.y + 1, 12)
                self.turnChar(heading, turn)
            elif heading == 11 and robot.y >= 1:
                robot.x = np.mod(robot.y - 1, 12)
                self.turnChar(heading, turn)
            elif heading == 0 and robot.y >= 1:
                robot.x = np.mod(robot.y - 1, 12)
                self.turnChar(heading, turn)
            elif heading == 1 and robot.y >= 1:
                robot.x = np.mod(robot.y - 1, 12)
                self.turnChar(heading, turn)
            else:
                self.turnChar(heading, turn)

        else: #not moving
            heading = heading
            robot.x = robot.x
            robot.y = robot.y

        if robot.x == robotX and robot.y == robotY:
            penalize = 0.0

        self.objects[0] = robot

        return penalize

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'robot':
                robot = obj
            else:
                others.append(obj)
        ended = False

        for other in others:
            if robot.x == other.x and robot.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gridObs(self.position('goal'), 1, 1, 1, 1, 'goal'))
                elif other.reward == -100:
                    self.objects.append(gridObs(self.position))