import numpy as np
import matplotlib.pyplot as plt

def st_ops(mu, q):
    x_prox = np.zeros(mu.shape)
    for i in range(len(mu)):
        if mu[i] > q:
            x_prox[i] = mu[i] - q
        else:
            if np.abs(mu[i]) < q:
                x_prox[i] = 0
            else:
                x_prox[i] = mu[i] + q
    return x_prox

A = np.array([[  3, 0.5],
              [0.5,   1]])
mu = np.array([[1],
               [2]])
L = 4 + np.sqrt(5)

epoch = 20
lams = [2, 4, 6]
w0_hist_lam = []
w1_hist_lam = []

for j, lam in enumerate(lams):
    w0_hist = np.zeros((epoch+1, 2))
    w1_hist = np.zeros((epoch+1, 2))
    wt = np.array([[3],
                   [-1]])
    for i in range(epoch):
        w0_hist[i] = wt[0]
        w1_hist[i] = wt[1]
        gradient = 2 * np.dot(A, wt-mu)
        wt = wt - 1/L * gradient
        wt = st_ops(wt, lam * 1/L)
    w0_hist[-1] = wt[0]
    w1_hist[-1] = wt[1]
    print(f'lam={lam}: w0={wt[0]}, w1={wt[1]}')

    for i in range(len(w0_hist)):
        w0_hist[i] = np.abs(w0_hist[i] - w0_hist[-1])
        w1_hist[i] = np.abs(w1_hist[i] - w1_hist[-1])
    w0_hist_lam.append(w0_hist)
    w1_hist_lam.append(w1_hist)

fig = plt.figure()
fig.subplots_adjust(wspace=0.4, hspace=0.6)
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.plot(w0_hist_lam[0], label='w0')
ax1.plot(w1_hist_lam[0], label='w1')
ax2.plot(w0_hist_lam[1], label='w0')
ax2.plot(w1_hist_lam[1], label='w1')
ax3.plot(w0_hist_lam[2], label='w0')
ax3.plot(w1_hist_lam[2], label='w1')
ax1.set_title('lam=2')
ax2.set_title('lam=4')
ax3.set_title('lam=6')
plt.show()
