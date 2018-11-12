import numpy as np
import math
import matplotlib.pyplot as plt

DT = 0.1 #time tick (s)
R = 20.
W = 85.
L_SPACE = 750.
W_SPACE = 500.
SIMTIME = 50. #simulation time (s)
MAXSPEED = 13.61 #rad/s

#MOTOR: FS90R - maximum rotation speed of around 130 RPM -> 130/9.5493 RPS = 13.61 Rad/s
#YAW -> Gyro Bias

class EKFmobile():
    def __init__(self):
        self.z = np.zeros((3, 1))
        # self.covR = np.diag([0.01, 0.01, math.radians(0.01), 0.01])**2 #Randomness in the state transition. Same dimension as the state vector 4
        # self.covR = np.diag([0.05, 0.05]) ** 2
        self.covR = np.diag([0.05, 0.05, 0.01, 0.01]) ** 2
        # X Y Yaw V_torso
        self.covQ = np.diag([0.01, 0.01, math.radians(0.01)]) ** 2 #measurement noise, k dimension same with measurement 3
        # Front_dist, Right_dist, Yaw
        self.wr = 0
        self.wl = 0

        # covariance for sensor output Laser Range Finders and IMU
        # Frond_dist, Right_dist, Yaw
        self.simQ = np.diag([0.01, 0.01, math.radians(0.01)])**2

        #covariance for input wr and wl
        self.covU = np.diag([0.05, 0.05]) ** 2

    def build_input(self, wr = 1, wl = 1):
        u = [wr, wl]
        self.wr = wr
        self.wl = wl
        return u


    # SHOULD DEAL WITH NOISE MODEL FIRST

    # def gyro_bias(self, yaw):
    #NEED to Get Response from DR. METHA


    def noise_propagation_time_evolution(self, u):
        #u is wr and wl

        #1. Add noise to input
        wr_noise = u[0] + ((MAXSPEED - np.random.normal(loc=MAXSPEED, scale=0.05)) * self.wr / MAXSPEED) * self.covR[0, 0]
        wl_noise = u[1] + ((MAXSPEED - np.random.normal(loc=MAXSPEED, scale=0.05)) * self.wl / MAXSPEED) * self.covR[1, 1]
        u_noise = [wr_noise, wl_noise]

        #2. Change to Velocity Term
        vel_torso_noise = R * (wr_noise + wl_noise) / 2 #Velocity of torso with process noise

        yawrate_noise = R * (wr_noise - wl_noise) / W

        return vel_torso_noise, yawrate_noise, u_noise



    def noise_propagation_h_observation(self, mu_true):
        # Q is used for measurements - Laser Range Finders, IMU
        # R is used for inputs - wr and wl
        # R - wr and wl have 5% standard deviation (maximum motor speed) -> need to change variance
        # 1) Front Distance by yaw
        # 2) Right Distance by yaw
        # 3) Yaw

        # 1. add noise to IMU - MPU-9250 and Laser Range Finders - VL53L0X - 3% STD
        x = mu_true[0]
        y = mu_true[1]
        yaw = mu_true[2]
        yaw = np.mod(yaw, 2 * np.pi)
        # yaw_rate = R * (wr - wl)/ W

        if yaw >= 0 and yaw < (np.pi / 2):  # 1st
            # (L_SPACE = 750., W_SPACE = 500.)
            z_front_dist = (W_SPACE - x) * np.cos(yaw) + (L_SPACE - y) * np.sin(yaw) + np.random.randn() * self.covQ[0, 0]
            z_right_dist = (W_SPACE - x) * np.sin(yaw) + y * np.cos(yaw) + np.random.randn() * self.covQ[1, 1]

        elif yaw >= (np.pi / 2) and yaw < (np.pi):  # 2nd
            z_front_dist = (L_SPACE - y) * np.sin(yaw) - x * np.cos(yaw) + np.random.randn() * self.covQ[0, 0]
            z_right_dist = (W_SPACE - x) * np.sin(yaw) - (L_SPACE - y) * np.cos(yaw) + np.random.randn() * self.covQ[1, 1]

        elif yaw >= np.pi and (yaw < (3 / 2) * np.pi):
            z_front_dist = -x * np.cos(yaw) - y * np.sin(yaw) + np.random.randn() * self.covQ[0, 0]
            z_right_dist = -(L_SPACE - y) * np.cos(yaw) - x * np.sin(yaw) + np.random.randn() * self.covQ[1, 1]

        elif yaw >= (3 / 2) * np.pi and yaw < (2 * np.pi):
            z_front_dist = -y * np.sin(yaw) + (W_SPACE - x) * np.cos(yaw) + np.random.randn() * self.covQ[0, 0]
            z_right_dist = -x * np.sin(yaw) + y * np.cos(yaw) + np.random.randn() * self.covQ[1, 1]

        z_yaw = yaw + np.random.randn() * self.covQ[2, 2]
        # print(np.diag([np.array(z_front_dist)[0][0], np.array(z_right_dist)[0][0], np.array(z_yaw)[0][0]]))

        self.z = np.array([z_front_dist[0], z_right_dist[0], z_yaw[0]])

        return self.z


    #g(u_t, mu_t-1)
    #state vector = [X, Y, yaw, V_torso]'
    #input vector = [w_right, w_left]'
    #TIME EVOLUTION MODEL
    def g_motion(self, mu, uReal):#mu_t_prediction = g(u_t, mu_t-1)

        # v, yawrate, u_noise = self.noise_propagation_time_evolution(u)
        v = R * (uReal[0] + uReal[1]) / 2
        yawrate = R * (uReal[0] - uReal[1]) / W

        # mu[2] = np.mod(mu[2], 2* np.pi)
        A = [[1, 0, 0, DT * np.cos(mu[2])], [0, 1, 0, DT * np.sin(mu[2])], [0, 0, 1, 0], [0, 0, 0, 1]]
        mu_pred = np.dot(A, mu)

        mu_pred[2] = mu_pred[2] + DT * yawrate
        mu_pred[3] = v

        #mu_pred is the result of system model
        return mu_pred

    def h_observation(self, mu_true):
        # Q is used for measurements - Laser Range Finders, IMU
        # R is used for inputs - wr and wl
        # R - wr and wl have 5% standard deviation (maximum motor speed) -> need to change variance
        # 1) Front Distance by yaw
        # 2) Right Distance by yaw
        # 3) Yaw

        # 1. add noise to IMU - MPU-9250 and Laser Range Finders - VL53L0X - 3% STD
        x = mu_true[0]
        y = mu_true[1]
        yaw = mu_true[2]
        yaw = np.mod(yaw, 2 * np.pi)
        # yaw_rate = R * (wr - wl)/ W

        if yaw >= 0 and yaw < (np.pi / 2):  # 1st
            # (L_SPACE = 750., W_SPACE = 500.)
            z_front_dist = (W_SPACE - x) * np.cos(yaw) + (L_SPACE - y) * np.sin(yaw)
            z_right_dist = (W_SPACE - x) * np.sin(yaw) + y * np.cos(yaw)

        elif yaw >= (np.pi / 2) and yaw < (np.pi):  # 2nd
            z_front_dist = (L_SPACE - y) * np.sin(yaw) - x * np.cos(yaw)
            z_right_dist = (W_SPACE - x) * np.sin(yaw) - (L_SPACE - y) * np.cos(yaw)

        elif yaw >= np.pi and (yaw < (3 / 2) * np.pi):
            z_front_dist = -x * np.cos(yaw) - y * np.sin(yaw)
            z_right_dist = -(L_SPACE - y) * np.cos(yaw) - x * np.sin(yaw)

        elif yaw >= (3 / 2) * np.pi and yaw < (2 * np.pi):
            z_front_dist = -y * np.sin(yaw) + (W_SPACE - x) * np.cos(yaw)
            z_right_dist = -x * np.sin(yaw) + y * np.cos(yaw)

        z_yaw = yaw + np.random.randn() * self.covQ[2, 2]
        # print(np.diag([np.array(z_front_dist)[0][0], np.array(z_right_dist)[0][0], np.array(z_yaw)[0][0]]))

        self.z = np.array([z_front_dist[0], z_right_dist[0], z_yaw[0]])

        return self.z



    def jacobianW(self, mu_pred, uReal):
        #Partial derivative in terms of noise v and noise w
        #4 by 2 matrix - [noise_wr, noise_wl].T
        #X = X + DT * cos(yaw) * ( V_torso + R * (nwr + nwl / 2))
        #Y = Y + DT * sin(yaw) * ( V_torso + R * (nwr + nwl / 2))
        #yaw = yaw + ( yawrate  + R * (nwr - nwl) / W ) * DT
        #V_torso = V_torso  +  R * (nwr + nwl / 2)_

        yaw = mu_pred[2]
        yaw = np.mod(yaw, 2 * np.pi)

        JW =[[(DT * np.cos(yaw) * R / 2).tolist()[0], (DT * np.cos(yaw) * R / 2).tolist()[0]],\
            [(DT * np.sin(yaw) * R / 2).tolist()[0], (DT * np.sin(yaw) * R / 2).tolist()[0]],\
            [R * DT / W, -R * DT / W],\
            [R/2, R/2]]

        return JW

    def jacobianG(self, mu_pred, uReal):
        #X = X + DT * cos(yaw) * ( V_torso + noise_v)
        #Y = Y + DT * sin(yaw) * ( V_torso + noise_v)
        #yaw = yaw + ( yawrate  + noise_W ) * DT
        #V_torso = V_torso  + noise_v

        yaw = mu_pred[2]
        yaw = np.mod(yaw, 2 * np.pi)
        v = R * (uReal[0] + uReal[1]) / 2

        JG =[[1.0, 0.0, (-DT * v * np.sin(yaw)).tolist()[0], (DT * np.cos(yaw)).tolist()[0]],\
            [0.0, 1.0, (DT * v * np.cos(yaw)).tolist()[0], (DT * np.sin(yaw)).tolist()[0]],\
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
                [-np.sin(yaw), np.cos(yaw), (W_SPACE - x) * np.cos(yaw) - y * np.sin(yaw), 0],\
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


    def EKF_process(self, mu, cov, u_noise, z_noise):
        #Before coming to EKF, u and z already had noise with noise propagation.
        #prediction - time update

        mu_pred = self.g_motion(mu, u_noise)
        #after g_motion, mu_pred also has noise part as well by u_noise.
        #so, mu_pred has noise part as well.

        JG = self.jacobianG(mu_pred, u_noise)
        JW = self.jacobianW(mu_pred, u_noise)
        # cov_pred = np.dot(np.dot(JG, cov), np.transpose(JG)) + np.dot(np.dot(JW, self.covR), np.transpose(JW)) #covariance of process noise - X, Y, yaw, V_torso, 4 by 4
        cov_pred = np.dot(np.dot(JG, cov), np.transpose(JG)) + self.covR  # covariance of process noise - X, Y, yaw, V_torso, 4 by 4
        # print(np.dot(np.dot(JW, self.covR), np.transpose(JW)))


        #update - observation update
        JH = self.jacobianH(mu_pred) #JH: 3 by 4


        z_pred = self.h_observation(mu_pred).reshape((3, ))
        y = (z_noise - z_pred).reshape((3, 1)) #3 by 1 matrix of z, innovation
        S = np.dot(np.dot(JH, cov_pred), np.transpose(JH)) + self.covQ #covariance of measurement noise - wr, wl
        K = np.dot(np.dot(cov_pred, np.transpose(JH)), np.linalg.inv(S))
        mu = mu_pred + np.dot(K, y)
        cov = np.dot((np.eye(len(mu)) - np.dot(K, JH)), cov_pred)


        print('mu: ', mu)


        return mu, cov


if __name__ == '__main__':
    print('start EKF process')

    time = 0.0

    #state vector [x y yaw v_torso]'
    mu = np.zeros((4, 1))
    mu_true = np.zeros((4, 1)) #[x, y, yaw, v]
    cov = np.eye(4)

    # stack space for mu, mu_true, z
    muList = mu
    mu_trueList = mu_true
    zList = np.zeros((1, 3))

    #class
    EKF = EKFmobile()

    while SIMTIME >= time:
        time += DT
        u = EKF.build_input(13, 13)

        _, _, ud = EKF.noise_propagation_time_evolution(u)
        z = EKF.noise_propagation_h_observation(mu_true)

        print('input: ', ud)


        mu, cov = EKF.EKF_process(mu, cov, ud, z)

        #save to space
        muList = np.hstack((muList, mu))
        mu_trueList = np.hstack((mu_trueList, mu_true))
        zList = np.vstack((zList, z))

        if True:
            plt.cla()
            # plt.plot(zList[:, 0], zList[:, 1], '.g')
            plt.plot(np.array(mu_trueList[0, :]).flatten(), np.array(mu_trueList[1, :]).flatten(), '-b')
            plt.plot(np.array(muList[0, :]).flatten(), np.array(muList[1, :]).flatten(), '-r')
            plt.axis("equal")
            plt.grid()
            plt.pause(0.001)


    # print(zList)











