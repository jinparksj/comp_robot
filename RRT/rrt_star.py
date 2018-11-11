"""

Path planning - Rapidly-exploring Random Trees* (RRT*)
Author: Sungjin Park

"""

import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib as mlp
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation

class RRTStar():
    def __init__(self, start, goal, size, obstacleList, onewayList, randArea, theta, expandDis = 1.0, goalSampleRate = 20, maxIter=600):
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
        self.maxIter = maxIter

    def planning(self, animation = True):
        self.nodeList = [self.start]
        for i in range(self.maxIter):
            if np.random.randint(0, 100) > self.goalSampleRate:
                rnd = [np.random.uniform(self.minrand, self.maxrand),\
                             np.random.uniform(self.minrand, self.maxrand), \
                             self.goal.theta]
            else:
                rnd = [self.goal.x, self.goal.y, self.goal.theta]

            min_ind = self.getNNindex(self.nodeList, rnd)
            # expand tree
            NN = self.nodeList[min_ind]
            self.theta = np.arctan2(rnd[1] - NN.y, rnd[0] - NN.x)

            newNode = copy.deepcopy(NN)
            newNode.x += self.expandDis * np.cos(self.theta)
            newNode.y += self.expandDis * np.sin(self.theta)
            newNode.theta = self.theta
            newNode.cost += self.expandDis
            newNode.parent = min_ind

            if self._CollisionCheck(newNode, self.obstacleList):
                nearIndices = self.find_near_nodes(newNode)
                newNode = self.choose_parent(newNode, nearIndices)
                self.nodeList.append(newNode)
                self.rewire(newNode, nearIndices)

            if not self._DirectionCheck(newNode, self.onewayList):
                continue

            if animation:
                self.DrawGraph(rnd)


            # dx = newNode.x - self.goal.x
            # dy = newNode.y - self.goal.y
            # d = np.sqrt(dx ** 2 + dy ** 2)
            # dangle = abs(newNode.theta - self.goal.theta)
            # if (d <= self.expandDis + self.sizeofrobot / 5) and dangle <= np.pi/7:
            #     print('robot is at the goal')
            #     break

        lastIndex = self.get_best_last_index()
        if lastIndex is None:
            return None

        path = self.get_final_course(lastIndex)
        return path



    def get_final_course(self, goalindex):
        path = [[self.goal.x, self.goal.y, self.goal.theta]]
        prenodex = self.goal.x
        prenodey = self.goal.y

        while self.nodeList[goalindex].parent is not None:
            node = self.nodeList[goalindex]
            node.theta = np.arctan2(prenodey - node.y, prenodex - node.x)
            path.append([node.x, node.y, node.theta])
            goalindex = node.parent

        path.append([self.start.x, self.start.y, self.start.theta])
        return path




    def get_best_last_index(self):
        disglist = np.array([[self.calc_dist_to_goal(node.x, node.y), node.theta] for node in self.nodeList])
        # goalindices = [disglist.index(i) for [i, theta] in disglist if (i <= (self.expandDis + self.sizeofrobot)) and (abs(theta - self.goal.theta <= np.pi/7))]
        goalindices = []
        for i, theta in disglist:
            if (i <= self.expandDis + self.sizeofrobot) and (abs(theta - self.goal.theta) <= np.pi / 5):
                goalindices.append(list(disglist[:, 0]).index(i))
        # goalindices = [disglist.index(i) for [i, theta] in disglist if
        #                (i <= (self.expandDis + self.sizeofrobot)) and (abs(theta - self.goal.theta <= np.pi / 7))]

        if len(goalindices) == 0:
            return None

        mincost = min([self.nodeList[i].cost for i in goalindices])
        for i in goalindices:
            if self.nodeList[i].cost == mincost:
                return i

        return None

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.goal.x, y - self.goal.y])



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


    def rewire(self, newNode, nearIndices):
        numbernode = len(self.nodeList)
        for i in nearIndices:
            NN = self.nodeList[i]
            dx = newNode.x - NN.x
            dy = newNode.y - NN.y
            d = np.sqrt(dx ** 2 + dy ** 2)

            scost = newNode.cost + d

            if NN.cost > scost:
                theta = np.arctan2(dy, dx)
                if self.check_collision_extend(NN, theta, d):
                    NN.parent = numbernode - 1
                    NN.cost = scost


    def choose_parent(self, newNode, nearIndices):
        if len(nearIndices) == 0:
            return newNode

        dlist = []

        for i in nearIndices:
            dx = newNode.x - self.nodeList[i].x
            dy = newNode.y - self.nodeList[i].y
            d = np.sqrt(dx ** 2 + dy ** 2)
            theta = np.arctan2(dy, dx)

            if self.check_collision_extend(self.nodeList[i], theta, d):
                dlist.append(self.nodeList[i].cost + d)
            else:
                dlist.append(float("inf"))
        mincost = min(dlist)
        minind = nearIndices[dlist.index(mincost)]

        if mincost == float("inf"):
            print("mincost is inf")
            return newNode

        newNode.cost = mincost
        newNode.parent = minind

        return newNode

    def check_collision_extend(self, nearNode, theta, d):
        tmpNode = copy.deepcopy(nearNode)

        for i in range(int(d / self.expandDis)):
            tmpNode.x += self.expandDis * np.cos(theta)
            tmpNode.y += self.expandDis * np.sin(theta)
            if not self._CollisionCheck(tmpNode, self.obstacleList):
                return False
        return True

    def find_near_nodes(self, newNode):
        numbernode = len(self.nodeList)
        coeff = self.expandDis * 50
        r = coeff * np.sqrt((np.log(numbernode) / numbernode))
        dlist = [(node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2 for node in self.nodeList]
        nearindices = [dlist.index(i) for i in dlist if i <= r ** 2]
        return nearindices

    def getNNindex(self, nodeList, rnd):
        nnlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in nodeList]
        min_ind = nnlist.index(min(nnlist))
        return min_ind

    def _CollisionCheck(self, node, obstacleList):
        for (ox, oy, size) in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = np.sqrt(dx ** 2 + dy ** 2)
            if d <= size + 1.5 * self.sizeofrobot: #collision case
                return False
        return True #avoid collision

    def _DirectionCheck(self, node, onewayList):
        for (ox, oy, size) in onewayList:
            dx = ox - node.x
            dy = oy - node.y
            d = np.sqrt(dx ** 2 + dy ** 2)
            if (d <= size + 1.5 * self.sizeofrobot) and \
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
        self.cost = 0.0
        self.parent = None


if __name__ == '__main__':
    print("RRT path planning")
    fig = plt.figure()

    obstacleList = [(300, 400, 100), (300, 500, 100), (300, 600, 100), (300, 700, 100), (300, 800, 100),
                    (300, 900, 100)]
    onewayList = [(100, 400, 100), (100, 500, 100), (100, 600, 100), (100, 700, 100), (100, 800, 100), (100, 900, 100),
                  (200, 400, 100), (200, 500, 100), (200, 600, 100), (200, 700, 100), (200, 800, 100), (200, 900, 100)]

    rrt = RRTStar(start=[115, 115, np.arctan2(600 - 115, 600 - 115)], goal=[600, 600, np.arctan2(600 - 115, 600 - 115)], \
              theta=0, size=115, randArea=[0, 700], obstacleList=obstacleList, onewayList=onewayList, expandDis=10, maxIter=800)
    path = rrt.planning(animation=True)


    rrt.DrawGraph()

    # plt.plot([x for (x, y) in smoothPath], [y for (x, y) in smoothPath], '-b')

    for (x, y, theta) in list(reversed(path)):
        print('(', x, y, theta, ')')
        # circular robot
        # plt.plot(x, y, '', ms = 115)
        rect_robot = mlp.patches.Rectangle((x - 42.5, y - 70), 85, 80, edgecolor='black', fill=False)
        rect_robot_center = mlp.patches.Rectangle((x - 42.5, y), 85, 10, edgecolor='red', fill=False)
        t = Affine2D().rotate_around(x, y, -theta)
        rect_robot.set_transform(t + plt.gca().transData)
        rect_robot_center.set_transform(t + plt.gca().transData)
        fig.add_subplot(111).add_patch(rect_robot)
        fig.add_subplot(111).add_patch(rect_robot_center)
        plt.pause(0.1)

    plt.grid(True)
    # animGIF = FuncAnimation(fig, main)
    # animGIF.save('rrt.gif', dpi=80, writer='imagemagick')
    plt.show()