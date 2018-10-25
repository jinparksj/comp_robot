import numpy as np
import math
import matplotlib.pyplot as plt

DT = 0.1 #time tick (s)
R = 20.
W = 85.
L_SPACE = 750.
W_SPACE = 500.
SIMTIME = 10. #simulation time (s)

#MOTOR: FS90R - maximum rotation speed of around 130 RPM -> 130/9.5493 RPS = 13.61 Rad/s


class EKFmobile():
    def __init__(self):
        self.z = np.zeros((3, 1))
        self.covR = np.diag([0.01, 0.01, math.radians(1.0), 0.01])**2 #Randomness in the state transition. Same dimension as the state vector 4
        # X Y Yaw V_torso
        self.covQ = np.diag([0.03, 0.03, math.radians(1.0)]) ** 2 #measurement noise, k dimension same with measurement 3
        # Front_dist, Right_dist, Yaw
        # self.wr = wr
        # self.wl = wl

        # covariance for sensor output Laser Range Finders and IMU
        # Frond_dist, Right_dist, Yaw
        self.simQ = np.diag([0.01, 0.01, math.radians(1.0)])**2

        #covariance for input wr and wl
        self.covU = np.diag([0.05, 0.05]) ** 2

    def build_input(self, wr = 1, wl = 1):
        u = [wr, wl]
        return u



    #g(u_t, mu_t-1)
    #state vector = [X, Y, yaw, V_torso]'
    #input vector = [w_right, w_left]'
    def g_motion(self, mu, u):#mu_t_prediction = g(u_t, mu_t-1)
        v = R * (u[0] + u[1]) / 2
        yawrate = R * (u[0] - u[1]) / W

        # mu[2] = np.mod(mu[2], 2* np.pi)
        A = [[1, 0, 0, DT * np.cos(mu[2])], [0, 1, 0, DT * np.sin(mu[2])], [0, 0, 1, 0], [0, 0, 0, 1]]
        mu_pred = np.dot(A, mu)

        mu_pred[2] = mu_pred[2] + DT * yawrate
        mu_pred[3] = v

        return mu_pred

    def h_observation(self, mu_pred): #h(mu_prediction)
        # mu_prediction = [X, Y, yaw, V_torso]'
        # measurement z:
        # 1) Laser range finders: distance to wall in a straight line(front), 2) distance to wall (right)
        # 3) IMU - angle wrt magnetic north, maybe
        # ????? 4) angular rate measurement - w of robot torso

        # 1) Front Distance by yaw
        # 2) Right Distance by yaw
        # 3) Yaw
        # ????? 4) Angular rate W_torso= r * (w_right - w_left) / W
        x = mu_pred[0]
        y = mu_pred[1]
        yaw = mu_pred[2]
        yaw = np.mod(yaw, 2*np.pi)
        # yaw_rate = R * (wr - wl)/ W

        if yaw >=0 and yaw < (np.pi / 2) : #1st
            #(L_SPACE = 750., W_SPACE = 500.)

            front_dist = (W_SPACE - x) * np.cos(yaw) + (L_SPACE - y) * np.sin(yaw)
            right_dist = (W_SPACE - x) * np.sin(yaw) + y * np.cos(yaw)


        elif yaw >= (np.pi/2) and yaw < (np.pi): #2nd
            front_dist = (L_SPACE - y) * np.sin(yaw) - x * np.cos(yaw)
            right_dist = (W_SPACE - x) * np.sin(yaw) - (L_SPACE - y) * np.cos(yaw)

        elif yaw >= np.pi and (yaw < (3/2) * np.pi):
            front_dist = -x * np.cos(yaw) - y * np.sin(yaw)
            right_dist = -(L_SPACE - y) * np.cos(yaw) - x * np.sin(yaw)

        elif yaw >= (3/2) * np.pi and yaw < (2 * np.pi):
            front_dist = -y * np.sin(yaw) + (W_SPACE - x) * np.cos(yaw)
            right_dist = -x * np.sin(yaw) + y * np.cos(yaw)

        self.z = np.array([front_dist, right_dist, yaw])

        return self.z


    def output_noise(self, mu_true, u):
        #Q is used for measurements - Laser Range Finders, IMU
        #R is used for inputs - wr and wl
        #R - wr and wl have 5% standard deviation (maximum motor speed) -> need to change variance
        # 1) Front Distance by yaw
        # 2) Right Distance by yaw
        # 3) Yaw

        mu_true = self.g_motion(mu_true, u)

        #1. add noise to IMU - MPU-9250 and Laser Range Finders - VL53L0X - 3% STD
        x = mu_true[0]
        y = mu_true[1]
        yaw = mu_true[2]
        yaw = np.mod(yaw, 2 * np.pi)
        # yaw_rate = R * (wr - wl)/ W

        if yaw >= 0 and yaw < (np.pi / 2):  # 1st
            # (L_SPACE = 750., W_SPACE = 500.)
            z_front_dist = (W_SPACE - x) * np.cos(yaw) + (L_SPACE - y) * np.sin(yaw) + np.random.randn() * self.simQ[0, 0]
            z_right_dist = (W_SPACE - x) * np.sin(yaw) + y * np.cos(yaw) + np.random.randn() * self.simQ[1, 1]

        elif yaw >= (np.pi / 2) and yaw < (np.pi):  # 2nd
            z_front_dist = (L_SPACE - y) * np.sin(yaw) - x * np.cos(yaw) + np.random.randn() * self.simQ[0, 0]
            z_right_dist = (W_SPACE - x) * np.sin(yaw) - (L_SPACE - y) * np.cos(yaw) + np.random.randn() * self.simQ[1, 1]

        elif yaw >= np.pi and (yaw < (3 / 2) * np.pi):
            z_front_dist = -x * np.cos(yaw) - y * np.sin(yaw) + np.random.randn() * self.simQ[0, 0]
            z_right_dist = -(L_SPACE - y) * np.cos(yaw) - x * np.sin(yaw) + np.random.randn() * self.simQ[1, 1]

        elif yaw >= (3 / 2) * np.pi and yaw < (2 * np.pi):
            z_front_dist = -y * np.sin(yaw) + (W_SPACE - x) * np.cos(yaw) + np.random.randn() * self.simQ[0, 0]
            z_right_dist = -x * np.sin(yaw) + y * np.cos(yaw) + np.random.randn() * self.simQ[1, 1]


        z_yaw = yaw + np.random.randn() * self.simQ[2, 2]
        # print(np.diag([np.array(z_front_dist)[0][0], np.array(z_right_dist)[0][0], np.array(z_yaw)[0][0]]))

        z = np.array([z_front_dist[0], z_right_dist[0], z_yaw[0]])

        #2. add noise to Input Motor Speed FS90R

        #WR
        ud1 = u[0] + np.random.randn() * self.covU[0, 0]
        ud2 = u[1] + np.random.randn() * self.covU[1, 1]
        ud = [ud1, ud2]

        return mu_true, z, ud



    def jacobianG(self, mu_pred, u):
        #X = X + DT * cos(yaw) * V_torso
        #Y = Y + DT * sin(yaw) * V_torso
        #yaw = yaw + yawrate * DT
        #V_torso = V_torso

        yaw = mu_pred[2]
        yaw = np.mod(yaw, 2 * np.pi)
        v = R * (u[0] + u[1]) / 2

        JG =[[1.0, 0.0, -DT * v * np.sin(yaw), DT * np.cos(yaw)],\
            [0.0, 1.0, DT * v * np.cos(yaw), DT * np.sin(yaw)],\
            [0.0, 0.0, 1.0, 0.0],\
            [0.0, 0.0, 0.0, 1.0]]

        return JG

    def jacobianH(self, mu_pred):
        # 3 by 4 matrix
        # 1) Front Distance by yaw
        # 2) Right Distance by yaw
        # 3) Yaw
        # ?????? 4) Angular rate W = r * (w_right - w_left) / W
        # [front_dist, right_dist, yaw, ang_rate]'

        x = mu_pred[0][0]
        y = mu_pred[1][0]
        yaw = mu_pred[2][0]
        yaw = np.mod(yaw, 2*np.pi)


        if yaw >=0 and yaw < (np.pi / 2) : #1st
            #(L_SPACE = 750., W_SPACE = 500.)

            # front_dist = (W_SPACE - x) * np.cos(yaw) + (L_SPACE - y) * np.sin(yaw)
            # right_dist = (W_SPACE - x) * np.sin(yaw) + y * np.cos(yaw)

            JH =[[-np.cos(yaw), -np.sin(yaw), -(W_SPACE - x) * np.sin(yaw) + (L_SPACE - y) * np.cos(yaw), 0],\
                [-np.sin(yaw), np.cos(yaw), (W_SPACE - x) * np.cos(yaw) - y * np.sin(yaw),0],\
                [0., 0., 1., 0.]]

            return JH


        elif yaw >= (np.pi/2) and yaw < (np.pi): #2nd
            # front_dist = (L_SPACE - y) * np.sin(yaw) - x * np.cos(yaw)
            # right_dist = (W_SPACE - x) * np.sin(yaw) - (L_SPACE - y) * np.cos(yaw)
            #
            JH =[[-np.cos(yaw), -np.sin(yaw), x * np.sin(yaw) + (L_SPACE - y) * np.cos(yaw), 0.],\
                [-np.sin(yaw), np.cos(yaw), (W_SPACE - x) * np.cos(yaw) + (L_SPACE - y) * np.sin(yaw),0.],\
                [0., 0., 1., 0.]]

            return JH

        elif yaw >= np.pi and (yaw < (3/2) * np.pi): #3rd
            # front_dist = -x * np.cos(yaw) - y * np.sin(yaw)
            # right_dist = -(L_SPACE - y) * np.cos(yaw) - x * np.sin(yaw)

            JH =[[-np.cos(yaw), -np.sin(yaw), x * np.sin(yaw) - y * np.cos(yaw), 0.],\
                [-np.sin(yaw), np.cos(yaw), - x * np.cos(yaw) + (L_SPACE - y) * np.sin(yaw),0.],\
                [0., 0., 1., 0.]]

            return JH


        elif yaw >= (3/2) * np.pi and yaw < (2 * np.pi):
            # front_dist = -y * np.sin(yaw) + (W_SPACE - x) * np.cos(yaw)
            # right_dist = -x * np.sin(yaw) + y * np.cos(yaw)

            JH =[[-np.cos(yaw), -np.sin(yaw), -(W_SPACE - x) * np.sin(yaw) - y * np.cos(yaw), 0.],\
                [-np.sin(yaw), np.cos(yaw), - x * np.cos(yaw) - y * np.sin(yaw),0.],\
                [0., 0., 1., 0.]]

            return JH


    def EKF_process(self, mu, cov, u, z):

        #prediction - time update
        mu_pred = self.g_motion(mu, u)
        JG = self.jacobianG(mu_pred, u)
        cov_pred = np.dot(np.dot(JG, cov), np.transpose(JG)) + self.covR #covariance of process noise - X, Y, yaw, V_torso, 4 by 4

        #update - observation update
        JH = self.jacobianH(mu_pred) #JH: 3 by 4


        z_pred = self.h_observation(mu_pred).reshape((3, ))
        y = (z - z_pred).reshape((3, 1)) #3 by 1 matrix of z, innovation
        S = np.dot(np.dot(JH, cov_pred), np.transpose(JH)) + self.covQ #covariance of measurement noise - wr, wl
        K = np.dot(np.dot(cov_pred, np.transpose(JH)), np.linalg.inv(S))
        mu = mu_pred + np.dot(K, y)
        cov = np.dot((np.eye(len(mu)) - np.dot(K, JH)), cov_pred)

        return mu, cov


if __name__ == '__main__':
    print('start EKF process')

    time = 0.0

    #state vector [x y yaw v_torso]'
    mu = np.zeros((4, 1))
    mu_true = np.zeros((4, 1))
    cov = np.eye(4)

    # stack space for mu, mu_true, z
    muList = mu
    mu_trueList = mu_true
    zList = np.zeros((1, 3))

    #class
    EKF = EKFmobile()

    while SIMTIME >= time:
        time += DT
        u = EKF.build_input(1, 1)


        mu_true, z, ud = EKF.output_noise(mu_true, u)


        mu, cov = EKF.EKF_process(mu, cov, ud, z)

        #save to space
        muList = np.hstack((muList, mu))
        mu_trueList = np.hstack((mu_trueList, mu_true))
        zList = np.vstack((zList, z))

        if True:
            plt.cla()
            plt.plot(zList[:, 0], zList[:, 1], '.g')
            plt.plot(np.array(mu_trueList[0, :]).flatten(), np.array(mu_trueList[1, :]).flatten(), '-b')
            plt.plot(np.array(muList[0, :]).flatten(), np.array(muList[1, :]).flatten(), '-r')
            plt.axis("equal")
            plt.grid()
            plt.pause(0.001)














