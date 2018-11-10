"""

Path planning - Rapidly-exploring Random Trees (RRT)
Author: Sungjin Park

"""

import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib as mlp
from matplotlib.transforms import Affine2D


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

            self.nodeList.append(newNode)
            # print("nNodeList: ", len(self.nodeList))
            # print("theta", self.theta)

            #check goal
            dx = newNode.x - self.goal.x
            dy = newNode.y - self.goal.y
            d = np.sqrt(dx ** 2 + dy ** 2)
            dangle = abs(newNode.theta - self.goal.theta)
            if (d <= self.expandDis + self.sizeofrobot / 5) and dangle <= np.pi/7:
                print('robot is at the goal')
                break



            if animation:
                self.DrawGraph(rnd)


        path = [[self.goal.x, self.goal.y, self.goal.theta]]
        lastIndex = len(self.nodeList) - 1
        while self.nodeList[lastIndex].parent is not None:
            node = self.nodeList[lastIndex]
            path.append([node.x , node.y, node.theta])
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
            plt.plot(ox, oy, "sk", ms = size)

        for (ox, oy, size) in self.onewayList:
            plt.plot(ox, oy, "sr", ms = size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "or", ms = 10)
        plt.axis([0, 700, 0, 1000])
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
            if d <= size + self.sizeofrobot: #collision case
                return False
        return True #avoid collision

    def _DirectionCheck(self, node, onewayList):
        for (ox, oy, size) in onewayList:
            dx = ox - node.x
            dy = oy - node.y
            d = np.sqrt(dx ** 2 + dy ** 2)
            if (d <= size + self.sizeofrobot) and \
                    not (node.theta <= np.rad2deg(315)\
                                 and node.theta >= np.rad2deg(225)): #direction check
                return False
        return True #avoid collision

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

        print(point1)
        print(point2)

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

if __name__ == "__main__":
    print("RRT path planning")
    fig = plt.figure()

    obstacleList = [(300, 400, 100), (300, 500, 100), (300, 600, 100), (300, 700, 100), (300, 800, 100), (300, 900, 100)]
    onewayList = [(100, 400, 100), (100, 500, 100), (100, 600, 100), (100, 700, 100), (100, 800, 100), (100, 900, 100),
        (200, 400, 100), (200, 500, 100), (200, 600, 100), (200, 700, 100), (200, 800, 100), (200, 900, 100)]

    rrt = RRT(start = [115, 115, np.arctan2(600-115, 600-115)], goal = [600, 600, np.arctan2(600-115, 600-115)], \
              theta=0, size = 115, randArea=[0, 700], obstacleList=obstacleList, onewayList=onewayList, expandDis= 10)
    path = rrt.planning(animation=True)


    #smoothing
    maxIter = 2000
    smoothPath = PathSmoothing(path, maxIter, obstacleList)

    rrt.DrawGraph()
    plt.plot([x for (x, y, theta) in smoothPath], [y for (x, y, theta) in smoothPath], '-b')
    # plt.plot([x for (x, y) in smoothPath], [y for (x, y) in smoothPath], '-b')

    for (x, y, theta) in list(reversed(smoothPath)):
        print('(', x, y, theta, ')')
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
    plt.show()


