import neo
from random import random, randomize
import times, os
var
  t = cpuTime()

randomize()

var x = randomMatrix(10000, 2, max = 3.0)

var y = zeros(10000, 1)

for i in 0..9999:
  y[i, 0] = x[i, 0] * 4 + 2 + random(0.2)

# let x_n = x.reshape(1000, 2)

for i in 0..9999:
  x[i,1] = 1.0

# For Polynomial
# for i in 0..999:
#   x[i,2] = x[1,0] ^ 2

# Batch
# var th = (x.t * x).inv * x.t * y
var theta = zeros(2,1)

# Add Stochastic and Mini Batch
# var eta = 0.1
var n_epoch = 1000
# var batch_size = 10
var m = 10000

proc learning_schedule(t: int): float =
    return 5 / (t + 50)

for epoch in 0..n_epoch:
  for n in 0..m:
#   Stochastic
    var rand = random(9999)
    var xi = zeros(2,2)
    # var xi = matrix(@[
    #   x.row(rand),
    #   x.row(rand+1)
    # ])
    # var xi = matrix(@[
    #   @[x[rand,0], x[rand,1]],
    #   @[x[rand+1,0], x[rand+1,1]],
    # ])
    # var yi = matrix(@[
    #   @[y[rand,0]],
    #   @[y[rand+1,0]]
    # ])
    xi[0,0] = x[rand, 0]
    xi[0,1] = x[rand, 1]
    xi[1,0] = x[rand+1, 0]
    xi[1,1] = x[rand+1, 1]
    var yi = zeros(2,1)
    yi[0,0] = y[rand, 0]
    yi[1,0] = y[rand+1, 0]
    var diff = xi.t * ((xi * theta) - yi)
    var eta = learning_schedule((epoch * m + n))
    theta = theta - diff * eta

# theta[0,0] = th[0,0]
# theta[1,0] = th[1,0]

echo theta

var x_n = matrix(
    @[
      @[0.0, 1.0],
      @[3.0, 1.0]
    ]
  )

var pred = x_n * theta

echo "x=0: ", pred[0,0]
echo "x=3: ", pred[1,0]

echo "Time taken: ",cpuTime() - t