#Extended Kalman Filter
'''

Authorized by Sungjin Park
UCLA, EE209AS
Computational Robotics

'''

'''
Notation summary

x, y: the position of robot
theta: the heading of robot
beta: yawrate * DT
alpha: steering angle

x_t+1 = x_t
y_t+1 = y_t
theta = theta + beta

state = [x, y, theta]
input = [wr, wl]
measurement = [front_dist, right_dist, theta(heading)]

1. mu = [x, y, theta]
2. x is the position of x
3. y is the position of y
4. theta is yaw
5. u is input
6. v is velocity of robot, v = R * (wr + wl) / 2
7. r, wr is the input of motor speed of right motor
8. l, wl is the input of motor speed of left motor
9. w is yawrate

'''


#LANDMARK Generation fxn
#Gyro Bias Term adding
#Variance from Datasheet

from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.stats import plot_covariance_ellipse, plot_covariance
from math import sqrt, tan, cos, sin, atan2
import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy.abc import mu, x, y, v, w, theta, u
from sympy import symbols, Matrix

DT = 1.0
STD_Wr = 0.05
STD_Wl = 0.05
STD_IMU = 0.00028
STD_RANGE = 0.03
R = 20
width = 85

sympy.init_printing(use_latex='mathjax', fontsize='12pt')

#2. Design measurement model

def JH_process():
    px, py = symbols('p_x, p_y')
    rx, ry = symbols('r_x, r_y')
    w, wb = symbols('w, wb')
    z = Matrix([[sympy.sqrt((px - x) ** 2 + (py - y) ** 2)],
                [sympy.sqrt((rx - x) ** 2 + (ry - y) ** 2)],
                [sympy.atan2(py - y, px - x) - theta],
                [w + wb]])
    JH = z.jacobian(Matrix([x, y, theta, w, wb]))
    return JH

print(JH_process())


def JH(mu, front_landmark_pos, right_landmark_pos):
    px = front_landmark_pos[0]
    py = front_landmark_pos[1]
    front_dist_sq = (px - mu[0, 0]) ** 2 + (py - mu[1, 0]) ** 2
    front_dist = np.sqrt((px - mu[0, 0]) ** 2 + (py - mu[1, 0]) ** 2)

    rx = right_landmark_pos[0]
    ry = right_landmark_pos[0]
    right_dist_sq = (rx - mu[0, 0]) ** 2 + (ry - mu[1, 0]) ** 2
    right_dist = np.sqrt(right_dist_sq)

    JH = np.array([[(-px + mu[0, 0]) / front_dist, (-py + mu[1, 0]) / front_dist, 0, 0, 0],
                   [(-rx + mu[0, 0]) / right_dist, (-ry + mu[1, 0]) / right_dist, 0, 0, 0],
                  [(py - mu[1, 0]) / front_dist_sq, (-px + mu[0, 0]) / front_dist_sq, -1, 0, 0],
                   [0, 0, 0, 1, 1]])

    return JH

#NEET to define function that converts the system state into a measurement
def Hfxn(mu, front_landmark_pos, right_landmark_pos):
    px = front_landmark_pos[0]
    py = front_landmark_pos[1]
    front_dist = np.sqrt((px - mu[0, 0]) ** 2 + (py - mu[1, 0]) ** 2)

    rx = right_landmark_pos[0]
    ry = right_landmark_pos[0]
    right_dist = np.sqrt((rx - mu[0, 0]) ** 2 + (ry - mu[1, 0]) ** 2)

    w = mu[3, 0]
    wb = mu[4, 0]

    Hfxn = np.array([[front_dist],
                     [right_dist],
                     [atan2(py - mu[1, 0], px - mu[0, 0]) - mu[2, 0]],
                     [w + wb]])
    return Hfxn

class EKFrobot(EKF):
    def __init__(self):
        #EKF Arguments: dimension of states, dim of measurements, dim of inputs
        EKF.__init__(self, dim_x= 5, dim_z=4, dim_u=2)
        self.dt = DT

        mu, x, y, v, w, theta, time, wb = symbols('mu, x, y, v, w, theta, t, wb')

        wr = symbols('wr')
        wl = symbols('wl')


        # 1. Design state model
        # fundamental matrix, F
        fMat = Matrix([[x + time * sympy.cos(theta) * (R * (wr + wl) / 2)],
                       [y + time * sympy.sin(theta) * (R * (wr + wl) / 2)],
                       [theta + time * (R * (wr - wl) / width) + time * wb],
                       [w + wb],
                       [wb]])

        self.JF = fMat.jacobian(Matrix([x, y, theta, w, wb]))  # jacobian F
        self.JW = fMat.jacobian(Matrix([wr, wl])) #jacobian W
        self.subs = {x: 0, y: 0, wr: 0, wl: 0, time: DT, theta: 0, wb: 0, w: 0}
        self.w = w
        self.wb = wb
        self.mu_x, self.mu_y = x, y
        self.theta = theta
        self.wr, self.wl = wr, wl


    def time_update(self, u = 0):
        self.mu = self.move(self.mu, u, self.dt)
        self.subs[self.theta] = self.mu[2, 0]
        self.subs[self.w] = self.mu[3, 0]
        self.subs[self.wb] = self.mu[4, 0]
        self.subs[self.wr] = u[0]
        self.subs[self.wl] = u[1]

        F = np.array(self.JF.evalf(subs=self.subs)).astype(float)
        W = np.array(self.JW.evalf(subs=self.subs)).astype(float)

        R = np.array([[STD_Wr * u[0] ** 2, 0], [0, STD_Wl * u[1] ** 2]])

        self.P = np.dot(F, self.P).dot(F.T) + np.dot(W, R).dot(W.T)


    def move(self, mu, u, DT):
        heading = mu[2, 0]
        vel = R * (u[0] + u[1]) / 2
        dist = vel * DT
        yawrate = R * (u[0] - u[1]) / width #yawrate = w
        yawchange = yawrate * DT
        w = yawrate
        wb = 0

        du = np.array([[dist * np.cos(heading)],
                       [dist * np.sin(heading)],
                       [yawchange],
                       [w],
                       [wb]])

        return mu + du

def residual_prcess(a, b):
    y = a - b
    y[1] = y[1] % (2 * np.pi)
    if y[1] > np.pi:
        y[1] -= 2 * np.pi
    return y

def z_landmark(front_landmark, right_landmark, sim_pos):
    x, y = sim_pos[0, 0], sim_pos[1, 0]
    w = sim_pos[3, 0]
    wb = sim_pos[4, 0]
    front_d = np.sqrt((front_landmark[0] - x) ** 2 + (front_landmark[1] - y) ** 2)
    right_d = np.sqrt((right_landmark[0] - x) ** 2 + (right_landmark[1] - y) ** 2)
    bearing = atan2(front_landmark[1] - y, front_landmark[0] - x) - sim_pos[2, 0]

    z = np.array([[front_d + np.random.randn() * STD_RANGE],
                  [right_d + np.random.randn() * STD_RANGE],
                  [bearing + np.random.randn() * STD_IMU],
                  [w + wb + np.random.randn() * STD_IMU]])
    return z

def ekf_update(ekf, z, front_landmark, right_landmark):
    ekf.update(z, HJacobian=JH, Hx=Hfxn, residual = residual_prcess,\
               args=(front_landmark, right_landmark), hx_args=(front_landmark, right_landmark))


def front_landmark_generation(sim_pos, yaw):
    slope = tan(yaw)
    x = sim_pos[0, 0]
    y = sim_pos[1, 0]

    if slope * (500 - x) + y <= 750 and slope * (500 - x) + y >= 0:
        landmark_y = slope * (500 - x) + y
        landmark = [500, landmark_y]
        return landmark

    elif slope * (0 - x) + y >= 0 and slope * (0 - x) + y <= 750:
        landmark_y = slope * (0 - x) + y
        landmark = [0, landmark_y]
        return landmark

    elif (1/slope) * (750 - y) + x <= 500 and (1/slope) * (750 - y) + x >= 0:
        landmark_x = (1/slope) * (750 - y) + x
        landmark = [landmark_x, 750]
        return landmark

    elif (1/slope) * (0 - y) + x <= 500 and (1/slope) * (0 - y) + x >= 0:
        landmark_x = (1/slope) * (0 - y) + x
        landmark = [landmark_x, 0]
        return landmark




def right_landmark_generation(sim_pos, yaw, front_landmark):
    slope = tan(yaw - np.deg2rad(90))
    x = sim_pos[0, 0]
    y = sim_pos[1, 0]

    fx = front_landmark[0]
    fy = front_landmark[1]

    if fx == 500:
        if (1 / slope) * (0 - y) + x <= 500 and (1 / slope) * (0 - y) + x >= 0: #ry = 0
            landmark_x = (1 / slope) * (0 - y) + x
            landmark = [landmark_x, 0]
            return landmark
        elif slope * (500 - x) + y <= 750 and slope * (500 - x) + y >= 0: #rx = 500
            landmark_y = slope * (500 - x) + y
            landmark = [500, landmark_y]
            return landmark
        elif slope * (0 - x) + y >= 0 and slope * (0 - x) + y <= 750: #rx = 0
            landmark_y = slope * (0 - x) + y
            landmark = [0, landmark_y]
            return landmark

    if fy == 750:
        if slope * (500 - x) + y <= 750 and slope * (500 - x) + y >= 0: #rx = 500
            landmark_y = slope * (500 - x) + y
            landmark = [500, landmark_y]
            return landmark
        elif (1 / slope) * (0 - y) + x <= 500 and (1 / slope) * (0 - y) + x >= 0: #ry = 0
            landmark_x = (1 / slope) * (0 - y) + x
            landmark = [landmark_x, 0]
            return landmark
        elif (1 / slope) * (750 - y) + x <= 500 and (1 / slope) * (750 - y) + x >= 0: #ry = 750
            landmark_x = (1 / slope) * (750 - y) + x
            landmark = [landmark_x, 750]
            return landmark

    if fx == 0:
        if (1 / slope) * (750 - y) + x <= 500 and (1 / slope) * (750 - y) + x >= 0: #ry = 750
            landmark_x = (1 / slope) * (750 - y) + x
            landmark = [landmark_x, 750]
            return landmark
        elif slope * (0 - x) + y >= 0 and slope * (0 - x) + y <= 750: #rx = 0
            landmark_y = slope * (0 - x) + y
            landmark = [0, landmark_y]
            return landmark
        elif slope * (500 - x) + y <= 750 and slope * (500 - x) + y >= 0: #rx = 500
            landmark_y = slope * (500 - x) + y
            landmark = [500, landmark_y]
            return landmark

    if fy == 0:
        if slope * (0 - x) + y >= 0 and slope * (0 - x) + y <= 750: #rx = 0
            landmark_y = slope * (0 - x) + y
            landmark = [0, landmark_y]
            return landmark
        elif (1 / slope) * (0 - y) + x <= 500 and (1 / slope) * (0 - y) + x >= 0: #ry = 0
            landmark_x = (1 / slope) * (0 - y) + x
            landmark = [landmark_x, 0]
            return landmark
        elif (1 / slope) * (750 - y) + x <= 500 and (1 / slope) * (750 - y) + x >= 0: #ry = 750
            landmark_x = (1 / slope) * (750 - y) + x
            landmark = [landmark_x, 750]
            return landmark




def run_localization(wr = 2, wl = 2, step = 10, ellipse_step = 20, ylim = None):
    ekf = EKFrobot()
    ekf.mu = np.array([[100, 100, .1, 0., 0.1]]).T #initialize states
    # randx = np.mod(np.random.randn() * 1000, 500)
    # randy = np.mod(np.random.randn() * 1000, 750)
    # randtheta = np.mod(np.random.randn() * 10, np.pi)
    # randw = np.mod(np.random.randn() * 10, 1)
    # randb = np.mod(np.random.randn() * 10, 1)
    # ekf.mu = np.array([[randx, randy, randtheta, randw, randb]]).T


    ekf.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.01]) #covariance matrix sigma
    ekf.R = np.diag([STD_RANGE ** 2, STD_RANGE ** 2, STD_IMU ** 2, STD_IMU ** 2])

    sim_pos = ekf.mu.copy()

    u = np.array([wr, wl])

    plt.figure()
    plt.title("EKF Wheeled Robot Localization")



    track = []

    for i in range(200):
        sim_pos = ekf.move(sim_pos, u, DT/10.)
        track.append(sim_pos)
        print(sim_pos)

        #NEED TO MAKE LANDMARK GENERATION!

        if i % step == 0:
            ekf.time_update(u = u)

            if i % ellipse_step == 0:
                plot_covariance((ekf.mu[0, 0], ekf.mu[1, 0]), ekf.P[0:4, 0:4], std=5, facecolor='r', alpha = 0.3)

            x, y = sim_pos[0, 0], sim_pos[1, 0]
            front_landmark = front_landmark_generation(sim_pos, ekf.mu[2, 0])
            plt.scatter(front_landmark[0], front_landmark[1], marker='s', s=20)
            right_landmark = right_landmark_generation(sim_pos, ekf.mu[2, 0], front_landmark)
            plt.scatter(right_landmark[0], right_landmark[1], marker='s', s=20)


            z = z_landmark(front_landmark, right_landmark, sim_pos)

            if z[0] <= 120 or z[1] <= 120:
                u = np.array([1, 0])

            if z[0] >= 100 and z[1] >= 100:
                u = np.array([1.1, 1])

            ekf_update(ekf, z, front_landmark, right_landmark)

            if i % ellipse_step == 0:
                plot_covariance((ekf.mu[0, 0], ekf.mu[1, 0]), ekf.P[0:4, 0:4], std=5, facecolor='k', alpha=0.8)




    track = np.array(track)
    plt.plot(track[:, 0], track[:, 1], color = 'b', lw = 2)
    # plt.axis('equal')
    plt.title("EKF Wheeled Robot Localization")
    plt.grid()
    plt.show()
    return ekf


ekf = run_localization(wr = 2.2, wl = 2)









