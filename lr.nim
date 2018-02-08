import neo
from random import random, randomize

randomize()

var x = randomMatrix(1000, 2, max = 3.0)

var y = x * 4

for i in 0..999:
  y[i, 0] = x[i, 0] * 4 + 2 + random(0.2)

# let x_n = x.reshape(1000, 2)

for i in 0..999:
  x[i,1] = 1.0

# For Polynomial
# for i in 0..999:
#   x[i,2] = x[1,0] ^ 2

# Batch
var th = (x.t * x).inv * x.t * y
var theta = zeros(2,1)

# Add Stochastic and Mini Batch

theta[0,0] = th[0,0]
theta[1,0] = th[1,0]

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