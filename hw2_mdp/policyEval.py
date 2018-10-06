import numpy as np

def policyEvaluation(policy, env, gamma = 1.0, theta = 0.00001):
    valList = np.zeros(env.size * env.size)
    while True: #full backup - > need to change one step backup
        delta = 0
        for s in range(env.size + env.size):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                pass
