"""

Path planning - Rapidly-exploring Random Trees (RRT)
Author: Sungjin Park

"""

import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib as mlp
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
import time

DT = 0.1

class RRT():
    def __init__(self, start, goal, size, obstacleList, onewayList, randArea, theta, expandDis = 1.0, goalSampleRate = 5):
        '''
        :param start: start position [x, y, theta]
        :param goal: goal position [x, y, theta]
        :param obstacleList: [[x, y, size],...]
        :param randArea: random smapling area [min, max]

        '''

        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.theta = theta
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.obstacleList = obstacleList
        self.sizeofrobot = size
        self.onewayList = onewayList
        # self.rightwayList = rightwayList

    def planning(self, animation = True):
        self.nodeList = [self.start]
        while True:
            if np.random.randint(0, 100) > self.goalSampleRate:
                rnd = [np.random.uniform(self.minrand, self.maxrand), np.random.uniform(self.minrand, self.maxrand), \
                       self.goal.theta]
            else:
                rnd = [self.goal.x, self.goal.y, self.goal.theta]

            #find NN
            minind = self.getNNindex(self.nodeList, rnd)
            #minind : [x, y, theta]

            #expand tree
            NN = self.nodeList[minind]
            self.theta = np.arctan2(rnd[1] - NN.y, rnd[0] - NN.x)

            newNode = copy.deepcopy(NN)
            newNode.x += self.expandDis * np.cos(self.theta)
            newNode.y += self.expandDis * np.sin(self.theta)
            newNode.theta = self.theta
            newNode.parent = minind

            if not self._CollisionCheck(newNode, self.obstacleList):
                continue

            if not self._DirectionCheck(newNode, self.onewayList):
                continue

            # if not self._RightDirectionCheck(newNode, self.rightwayList):
            #     continue

            self.nodeList.append(newNode)

            #check goal
            dx = newNode.x - self.goal.x
            dy = newNode.y - self.goal.y
            d = np.sqrt(dx ** 2 + dy ** 2)
            dangle = abs(newNode.theta - self.goal.theta)
            if (d <= self.expandDis + self.sizeofrobot / 5) and dangle <= np.pi/4:
                print('robot is at the goal')
                break



            if animation:
                self.DrawGraph(rnd)


        path = [[self.goal.x, self.goal.y, self.goal.theta]]
        lastIndex = len(self.nodeList) - 1
        prenodex = self.goal.x
        prenodey = self.goal.y
        while self.nodeList[lastIndex].parent is not None:
            node = self.nodeList[lastIndex]
            node.theta = np.arctan2(prenodey - node.y, prenodex - node.x)
            path.append([node.x, node.y, node.theta])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y, self.start.theta])

        return path

    def DrawGraph(self, rnd = None):
        plt.clf()

        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")

        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "sk", ms = 0.8 * size)

        for (ox, oy, size) in self.onewayList:
            plt.plot(ox, oy, "sr", ms = size)

        # for (ox, oy, size) in self.rightwayList:
        #     plt.plot(ox, oy, "s", ms = size, markerfacecolor = "None", markeredgecolor='blue', markeredgewidth='5')


        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "or", ms = 10)
        plt.axis([0, 800, 0, 1000])
        plt.grid(True)
        plt.pause(0.0001)




    def getNNindex(self, nodeList, rnd):
        nnlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in nodeList]
        min_ind = nnlist.index(min(nnlist))
        return min_ind

    def _CollisionCheck(self, node, obstacleList):
        for (ox, oy, size) in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = np.sqrt(dx ** 2 + dy ** 2)
            if d <= size + self.sizeofrobot * 1.2: #collision case
                return False
        return True #avoid collision

    def _DirectionCheck(self, node, onewayList):
        for (ox, oy, size) in onewayList:
            dx = ox - node.x
            dy = oy - node.y
            d = np.sqrt(dx ** 2 + dy ** 2)
            check_theta = np.arctan2(node.y - self.start.y, node.x - self.start.x)
            if (d <= size + self.sizeofrobot * 1.2) and \
                    not (check_theta <= np.deg2rad(315)\
                                 and check_theta >= np.deg2rad(225)): #direction check
                return False
        return True #avoid collision

    # def _RightDirectionCheck(self, node, rightwayList):
    #     for (ox, oy, size) in rightwayList:
    #         dx = ox - node.x
    #         dy = oy - node.y
    #         d = np.sqrt(dx ** 2 + dy ** 2)
    #         right_theta = np.arctan2(node.y - self.start.y, node.x - self.start.x)
    #         if (d <= size + self.sizeofrobot) and \
    #                 (right_theta <= np.deg2rad(270)\
    #                              and right_theta >= np.deg2rad(90)): #direction check
    #             return False
    #     return True


class Node():
    '''
    RRT Node
    '''

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None

def PathSmoothing(path, maxIter, obstacleList):
    length = GetPathLength(path)

    for i in range(maxIter):
        #sample 2 points
        pickPoints = [np.random.uniform(0, length), np.random.uniform(0, length)]
        pickPoints.sort()

        point1 = GetTargetPoint(path, pickPoints[0])
        point2 = GetTargetPoint(path, pickPoints[1])

        if (point1[3] <= 0) or (point2[3] <= 0):
            continue

        if (point2[3] + 1) > len(path):
            continue

        if point2[3] == point1[3]:
            continue

        if not lineCollisionCheck(point1, point2, obstacleList):
            continue

        newPath = []
        newPath.extend(path[:point1[3] + 1])
        newPath.append([point1[0], point1[1], point1[2]])
        newPath.append([point2[0], point2[1], point2[2]])
        newPath.extend(path[point2[3] + 1: ])
        path = newPath
        length = GetPathLength(path)

    return path

def GetPathLength(path):
    length = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = np.sqrt(dx ** 2 + dy ** 2)
        length += d
    return length

def GetTargetPoint(path, targetL):
    length = 0
    targetInd = 0
    lastPairLength = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = np.sqrt(dx ** 2 + dy ** 2)
        length += d
        if length >= targetL:
            targetInd = i - 1
            lastPairLength = d
            break

    partRatio = (length - targetL) / lastPairLength

    x = path[targetInd][0] + (path[targetInd + 1][0] - path[targetInd][0]) * partRatio
    y = path[targetInd][1] + (path[targetInd + 1][1] - path[targetInd][1]) * partRatio
    theta = path[targetInd][2] + (path[targetInd + 1][2] - path[targetInd][2]) * partRatio
    return [x, y, theta, targetInd]


def lineCollisionCheck(point1, point2, obstacleList):
    x1 = point1[0]
    y1 = point1[1]

    x2 = point2[0]
    y2 = point2[1]

    try:
        a = y2 - y1
        b = -(x2 - x1)
        c = y2 * (x2 - x1) - x2 * (y2 -y1)

    except ZeroDivisionError:
        return False

    for (ox, oy, size) in obstacleList:
        d = abs(a * ox + b * oy + c) / (np.sqrt(a ** 2 + b ** 2))
        if d <= size + 115:
            return False

    return True


class Controller():
    def __init__(self, velx, vely, angvel, startpos, thetaList, velList): #path has x, y, theta
        self.velx = velx
        self.vely = vely
        self.angvel = angvel
        self.start_pos = startpos
        self.dt = 1
        self.theta = thetaList
        self.vel = velList

    def move(self):
        temp_x = self.start_pos[0]
        temp_y = self.start_pos[1]
        temp_theta = self.start_pos[2]

        velxList = self.velx
        velyList = self.vely
        angvelList = self.angvel
        velList = self.vel

        uList = [self.start_pos]
        j = 0
        for i in range(len(velxList)):
            dist_x = velxList[i] * self.dt
            dist_y = velyList[i] * self.dt
            dist = velList[i] * self.dt
            thetachange = angvelList[i] * self.dt
            cur_theta = temp_theta + thetachange
            if np.mod(i, 10) == 0:
                prct_theta = thetaList[j]
                j += 1
            cur_x = temp_x + dist * np.cos(thetachange)
            cur_y = temp_y + dist * np.sin(thetachange)



            uList.append([cur_x, cur_y, cur_theta])

            temp_x = cur_x
            temp_y = cur_y
            temp_theta = cur_theta



        return uList



def convert_path_to_desired_input(smoothpath): #total 1 second to get goal path
    total_step = len(smoothpath)
    dt = 1
    np_path = np.array(smoothpath)
    xList = list(np_path[:, 0])
    yList = list(np_path[:, 1])
    thetaList = list(np_path[:, 2])

    extxList = []
    extyList = []
    extthetaList = []
    distList = []
    dthetaList = []

    angvelList = []

    velxList = []
    velyList = []
    velList = []


    for i in range(total_step - 1):
        dx = (xList[i + 1] - xList[i]) * dt
        dy = (yList[i + 1] - yList[i]) * dt
        ddist = np.sqrt((xList[i + 1] - xList[i]) ** 2 + (yList[i + 1] - yList[i]) ** 2) * dt
        dtheta = (thetaList[i + 1] - thetaList[i]) * dt
        for j in range(int(1/dt)):
            extxList.append(xList[i] + j * dx)
            extyList.append(yList[i] + j * dy)
            extthetaList.append(thetaList[i] + j * dtheta)

            distList.append(ddist)
            dthetaList.append(dtheta)
            # vel_x = dx / (dt * np.cos(thetaList[i]))
            # vel_y = dy / (dt * np.sin(thetaList[i]))
            vel_x = dx / dt
            vel_y = dy / dt
            vel = vel_x * np.cos(thetaList[i]) + vel_y * np.sin(thetaList[i])
            # vel = ddist / dt
            # ang_vel = dtheta / dt
            # vel = dx / (0.01 * np.sin(thetaList[i]))
            ang_vel = dtheta / dt
            angvelList.append(ang_vel)
            velxList.append(vel_x)
            velyList.append(vel_y)
            velList.append(vel)

    return velxList, velyList, angvelList, thetaList, velList, xList, yList


if __name__ == "__main__":
    starttime = time.time()
    # fig = plt.figure()
    print("RRT path planning")
    fig = plt.figure()


    obstacleList = [(300, 600, 200), (800, 100, 200)]
    onewayList = [(100, 600, 200), (100, 700, 200), (100, 800, 200), (100, 900, 200),
                  (200, 600, 200), (200, 700, 200), (200, 800, 200), (200, 900, 200)]

    # rightwayList = [(100, 100, 200), (200, 100, 200), (300, 100, 200)]

    print("planner working")
    rrt = RRT(start = [115, 115, np.arctan2(700-115, 700-115)], goal = [700, 700, np.arctan2(700-115, 700-115)], \
              theta=0, size = 115, randArea=[0, 800], obstacleList=obstacleList, onewayList=onewayList, expandDis=10)
    path = rrt.planning(animation=True)

    print("smoothing trajectory")
    #smoothing
    maxIter = 1000
    smoothPath = PathSmoothing(path, maxIter, obstacleList)

    print('smoothpath: ', list(reversed(smoothPath)))

    rrt.DrawGraph()
    plt.plot([x for (x, y, theta) in smoothPath], [y for (x, y, theta) in smoothPath], '-b')

    velx, vely, angvel, thetaList, velList, xList, yList = convert_path_to_desired_input(list(reversed(smoothPath)))

    print('velocity: ', velList)
    print('angular velocity: ', angvel)
    startpos = [115, 115, np.arctan2(700 - 115, 700 - 115)]

    ctrl = Controller(velx, vely, angvel, startpos, thetaList, velList)
    uList = ctrl.move()

    # print('final state: ', uList)


    for (x, y, theta) in list(reversed(smoothPath)):
    # for (x, y, theta) in uList:
        # print('(', x, y, theta, ')')
        #circular robot
        # plt.plot(x, y, '', ms = 115)
        rect_robot = mlp.patches.Rectangle((x - 42.5, y - 70), 85, 80, edgecolor = 'black', fill=False)
        rect_robot_center = mlp.patches.Rectangle((x - 42.5, y), 85, 10, edgecolor = 'red', fill=False)
        t = Affine2D().rotate_around(x, y, -theta)
        rect_robot.set_transform(t + plt.gca().transData)
        rect_robot_center.set_transform(t + plt.gca().transData)
        fig.add_subplot(111).add_patch(rect_robot)
        fig.add_subplot(111).add_patch(rect_robot_center)
        plt.pause(0.01)

    plt.grid(True)
    # animGIF = FuncAnimation(fig, main)
    # animGIF.save('rrt.gif', dpi=80, writer='imagemagick')
    endtime = time.time()
    operatingtime = endtime - starttime
    print('operating time(s): ', operatingtime)

    totaldist = 0
    for i in range(len(xList)-1):
        dist = np.sqrt((xList[i + 1] - xList[i]) ** 2 + (yList[i + 1] - yList[i]) ** 2)
        totaldist += dist

    print('trajectory efficiency(mm): ', totaldist)
    plt.show()


