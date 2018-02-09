import numpy as np
import time

start = time.time()

x = 2 * np.random.rand(10000, 1) * 3
y = 2 + 4 * x + (np.random.rand(10000, 1) * .2)

x_n = np.c_[np.ones(x.shape), x]

theta = np.random.rand(2, 1)
eta = .1
n_epoch = 1000
batch_size = 10
m = len(x_n)
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

diff = np.array([[1],[1]])
steps = 0
ep = 0
for epoch in range(n_epoch):
  for n in range(m):
#   Stochastic
    rand = np.random.randint(0,9999)
    xi, yi = x_n[rand:rand+1], y[rand:rand+1]
    diff = xi.T @ ((xi @ theta) - yi)
    eta = learning_schedule(epoch * m + n)
    theta = theta - diff * eta

print(theta)

x_t = np.array([[0],[3]])

print(np.c_[np.ones(x_t.shape), x_t] @ theta)

print(time.time() - start)