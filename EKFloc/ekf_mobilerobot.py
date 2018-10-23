import numpy as np
import math
import matplotlib.pyplot as plt

DT = 0.1
R = 20.
W = 85.
L_SPACE = 750.
W_SPACE = 500.



class EKFmobile():
    def __init__(self):
        self.z = np.zeros((4, 1))
        # self.wr = wr
        # self.wl = wl

    def build_input(self, wr, wl):
        u = np.matrix([wr, wl]).T
        return u

    #g(u_t, mu_t-1)
    #state vector = [X, Y, yaw, V_torso]'
    #input vector = [w_right, w_left]'
    def g_motion(self, mu, u):#mu_t_prediction = g(u_t, mu_t-1)
        v = R * (u[0] + u[1]) / 2
        yawrate = R * (u[0] - u[1]) / W

        # mu[2] = np.mod(mu[2], 2* np.pi)
        A = np.matrix([[1, 0, 0, DT * np.cos(mu[2])], [0, 1, 0, DT * np.sin(mu[2])], [0, 0, 1, 0], [0, 0, 0, 1]])
        mu_pred = A * mu

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


    def add_noise(self, u):
        #Q is used for measurements - Laser Range Finders, IMU
        #R is used for inputs - wr and wl
        #R - wr and wl have 5% standard deviation (maximum motor speed) -> need to change variance
        pass


    def jacobianG(self, mu_pred, u):
        #X = X + DT * cos(yaw) * V_torso
        #Y = Y + DT * sin(yaw) * V_torso
        #yaw = yaw + yawrate * DT
        #V_torso = V_torso

        yaw = mu_pred[2]
        yaw = np.mod(yaw, 2 * np.pi)
        v = R * (u[0] + u[1]) / 2

        JG = np.matrix([
            [1.0, 0.0, -DT * v * np.sin(yaw), DT * np.cos(yaw)],
            [0.0, 1.0, DT * v * np.cos(yaw), DT * np.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])

        return JG

    def jacobianH(self, mu_pred):
        # 3 by 4 matrix
        # 1) Front Distance by yaw
        # 2) Right Distance by yaw
        # 3) Yaw
        # ?????? 4) Angular rate W = r * (w_right - w_left) / W
        # [front_dist, right_dist, yaw, ang_rate]'

        x = mu_pred[0]
        y = mu_pred[1]
        yaw = mu_pred[2]
        yaw = np.mod(yaw, 2*np.pi)


        if yaw >=0 and yaw < (np.pi / 2) : #1st
            #(L_SPACE = 750., W_SPACE = 500.)

            # front_dist = (W_SPACE - x) * np.cos(yaw) + (L_SPACE - y) * np.sin(yaw)
            # right_dist = (W_SPACE - x) * np.sin(yaw) + y * np.cos(yaw)

            JH = np.matrix([
                [-np.cos(yaw), -np.sin(yaw), -(W_SPACE - x) * np.sin(yaw) + (L_SPACE - y) * np.cos(yaw), 0],
                [-np.sin(yaw), np.cos(yaw), (W_SPACE - x) * np.cos(yaw) - y * np.sin(yaw),0],
                [0., 0., 1., 0.]])

            return JH


        elif yaw >= (np.pi/2) and yaw < (np.pi): #2nd
            # front_dist = (L_SPACE - y) * np.sin(yaw) - x * np.cos(yaw)
            # right_dist = (W_SPACE - x) * np.sin(yaw) - (L_SPACE - y) * np.cos(yaw)
            #
            JH = np.matrix([
                [-np.cos(yaw), -np.sin(yaw), x * np.sin(yaw) + (L_SPACE - y) * np.cos(yaw), 0.],
                [-np.sin(yaw), np.cos(yaw), (W_SPACE - x) * np.cos(yaw) + (L_SPACE - y) * np.sin(yaw),0.],
                [0., 0., 1., 0.]])

            return JH

        elif yaw >= np.pi and (yaw < (3/2) * np.pi): #3rd
            # front_dist = -x * np.cos(yaw) - y * np.sin(yaw)
            # right_dist = -(L_SPACE - y) * np.cos(yaw) - x * np.sin(yaw)

            JH = np.matrix([
                [-np.cos(yaw), -np.sin(yaw), x * np.sin(yaw) - y * np.cos(yaw), 0.],
                [-np.sin(yaw), np.cos(yaw), - x * np.cos(yaw) + (L_SPACE - y) * np.sin(yaw),0.],
                [0., 0., 1., 0.]])

            return JH


        elif yaw >= (3/2) * np.pi and yaw < (2 * np.pi):
            # front_dist = -y * np.sin(yaw) + (W_SPACE - x) * np.cos(yaw)
            # right_dist = -x * np.sin(yaw) + y * np.cos(yaw)

            JH = np.matrix([
                [-np.cos(yaw), -np.sin(yaw), -(W_SPACE - x) * np.sin(yaw) - y * np.cos(yaw), 0.],
                [-np.sin(yaw), np.cos(yaw), - x * np.cos(yaw) - y * np.sin(yaw),0.],
                [0., 0., 1., 0.]])

            return JH












