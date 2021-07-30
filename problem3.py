import numpy as np
import matplotlib.pyplot as plt

def score(alpha, x, y):
    lam = 0.01
    w = np.zeros(x.shape[1])

    for a_i, x_i, y_i in zip(alpha, x, y):
        w += a_i * y_i * x_i
    w /= 2 * lam
    score = lam * np.dot(w, w)
    for x_i, y_i in zip(x, y):
        score += max(0, 1-y_i*np.dot(w.T, x_i)) / x.shape[1]
    return score

def getGradient(alpha, x, y):
    lam = 0.01
    K = np.dot(x, x.T) * np.outer(y, y)
    gradient = np.dot(K, alpha) / (2*lam) - np.ones(alpha.shape)
    return gradient

def SGM(x, y):
    eta = 0.0001
    epoch = 50
    alpha = np.random.rand(x.shape[0])
    score_hist = np.zeros(epoch)
    for i in range(epoch):
        alpha -= eta * getGradient(alpha, x, y)
        for j in range(len(alpha)):
            if alpha[j] > 1:
                alpha[j] = 1
            elif alpha[j] < 0:
                alpha[j] = 0
        score_hist[i] = score(alpha, x, y)
        print(f'{i}: {score_hist[i]}')
    return score_hist

n = 200
np.random.seed(seed=512)
omega = np.random.randn()
noise = 0.8 * np.random.randn(n)

x_d2 = np.random.randn(n, 2) + 0
y_d2 = 2 * (omega * x_d2[:,0] + x_d2[:,1] + noise > 0) - 1

score_hist = SGM(x_d2, y_d2)
plt.plot(score_hist)
plt.show()
