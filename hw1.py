import numpy as np
from control import *
import scipy as sp
from scipy.signal import TransferFunction
import matplotlib.pyplot as plt


# matA = np.mat([[1, 2, 3, 2], [3, 6, 9, 5], [2, 4, 6, 9]])
# print(np.rank(matA))
#
#
# num = [1, 2]
# den = [1, -1, -2]
# tf_2 = tf(num, den)
#
# sys = TransferFunction(num, den)
# print(sys)
# sys_ss = sys.to_ss()
# print(sys_ss)

resultSeq10 = [[]] * 10
numH = 0
numT = 0
likBiased = 0.25
likFair = 0.75
lik_T_bias = 0.75
lik_H_bias = 0.25
lik_T_fair = 0.5
lik_H_fair = 0.5


def coinFlip(type, numFlips, numH, numT):
    resultFlip = []
    if type == 'fair':
        for i in range(numFlips):
            coin = np.random.random_sample(1)
            if coin >= 0.5:
                resultFlip.append('H')
                numH = numH + 1
            else:
                resultFlip.append('T')
                numT = numT + 1
    elif type == 'biased':
        for i in range(numFlips):
            coin = np.random.random_sample(1)
            if coin <= 0.25:
                resultFlip.append('H')
                numH = numH + 1
            else:
                resultFlip.append('T')
                numT = numT + 1

    prob_seq = (likBiased * (lik_H_bias ** numH) * (lik_T_bias ** numT)) / \
            (likFair * (lik_H_fair ** (numH + numT)) + likBiased * (lik_H_bias ** numH) * (lik_T_bias ** numT))

    return resultFlip, prob_seq, numH, numT

# 4.(a) and (b)
for i in range(10):

    if i <= 4:
        flipFair, prob_seq, numH, numT = coinFlip('fair', 40, numH, numT)
        resultSeq10[i].append(flipFair)
    else:
        flipBiased, prob_seq, numH, numT = coinFlip('biased', 40, numH, numT)
        resultSeq10[i].append(flipFair)

#4. (c)

resultSeq10 = [[]] * 10

plt.figure(1)
prob_seqList = []
for j in range(5):
    numH = 0
    numT = 0
    for i in range(100):
        flipFair, prob_seq, numH, numT = coinFlip('fair', 1, numH, numT)
        prob_seqList.append(prob_seq)
    plt.plot(prob_seqList, label = (j+1))
    prob_seqList = []

plt.xlabel('number of flipping')
plt.ylabel('likelihood')
plt.title('4. (c) fair coin')
plt.legend()
plt.grid()


#4. (d)

resultSeq10 = [[]] * 10

plt.figure(2)
prob_seqList = []
for j in range(5):
    numH = 0
    numT = 0
    for i in range(100):
        flipFair, prob_seq, numH, numT = coinFlip('biased', 1, numH, numT)
        prob_seqList.append(prob_seq)
    plt.plot(prob_seqList, label = (j+1))
    prob_seqList = []

plt.xlabel('number of flipping')
plt.ylabel('likelihood')
plt.title('4. (d) biased coin')
plt.legend()
plt.grid()

plt.show()






