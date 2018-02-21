import neo
import strutils
from random import random, randomize, shuffle
import times, os, parsecsv, streams, math

# Reads csv file to matrix ("filepath", n_rows, n_cols) : how can it detect n_rows and n_cols?
proc read_csv(fn: string, rows: int, cols: int): Matrix[float64] =
  var x = zeros(rows, cols, float64)
  var s = newFileStream(fn, fmRead)
  if s == nil: quit("cannot open the file" & paramStr(1))
  var csv: CsvParser
  open(csv, s, fn)
  var row = 0
  var col = 0
  while readRow(csv):
    col = 0
    for val in items(csv.row):
      x[row, col] = parseFloat(val)
      col += 1
    row += 1
  close(csv)
  return x

# Get the shape of a matrix
proc shape(X: Matrix[float64]): (int, int) =
  var c = 0
  var r = 0
  for val in X.column(0):
    r+=1
  for val in X.row(0):
    c+=1
  return (r, c)

# Splits X and y into training and testing data : ts is test set size between 0 - 1, seed is seed for randomize
proc train_test_split(X: Matrix[float64], y: Matrix[float64], ts: float64, seed: int): (Matrix[float64], Matrix[float64], Matrix[float64], Matrix[float64]) =
  # create random indicies
  randomize(seed)

  var
    (rows, cols) = shape(X)
    ind = newSeq[int](rows-1)
  for i in 0..rows-2: ind[i] = i
  shuffle(ind)

  var
     train_size = int(round(float(rows) * (1-ts)))
     test_size = int(round(float(rows) * ts))
     x_train = zeros(train_size, cols)
     x_test = zeros(test_size, cols)
     y_train = zeros(train_size, 1)
     y_test = zeros(test_size, 1)
     count = 0

  for col in 0..cols-1:
    for row in ind:
      if count < train_size:
        x_train[count, col] = X[row, col]
        y_train[count, 0] = y[row, 0]
      else:
        if count-train_size < test_size:
          x_test[(count-train_size), col] = X[row, col]
          y_test[(count-train_size), 0] = y[row, 0]
      count+=1
      
  return (x_train, x_test, y_train, y_test)

var train = read_csv("./data/iris_training.csv", 120, 5)

var test = read_csv("./data/iris_test.csv", 30, 5)

# Drop a column from matrix
proc drop_column(X: Matrix[float64], drop: int): Matrix[float64] =
  var (rows, cols) = shape(X)
  var nX = zeros(rows, cols-1)
  var newCol = 0
  for col in 0..cols-1:
    if col != drop:
      for row in 0..rows-1:
        nX[row, newCol] = X[row, col]
      newCol+=1
  return nX

proc get_column(X: Matrix[float64], get: int): Matrix[float64] =
  var (rows, cols) = shape(X)
  return X.column(get).asMatrix(rows, 1)

var y_train = get_column(train, 4)
var x_train = drop_column(train, 4)

var (xt, yt, xts, yts) = train_test_split(x_train, y_train, 0.25, 29)

var y_test = get_column(test, 4)
var x_test = drop_column(test, 4)

# Normalize matrix columns
proc normalize(X: Matrix[float64]): Matrix[float64] =
  var (rows, cols) = shape(X)
  var nX = zeros(rows, cols)
  for col in 0..cols-1:
    for row in 0..rows-1:
      nX[row, col] = X[row, col] / max(X.column(col))
  return nX

# Adds a column of 1s to a matrix
proc add_ones(X: Matrix[float64]): Matrix[float64] =
  var (rows, cols) = shape(X)
  var nX = zeros(rows, cols+1)
  for col in 0..cols:
    if col == cols:
      for row in 0..rows-1:
        nX[row, col] = 1 
    else:   
      for row in 0..rows-1:
        nX[row, col] = X[row, col]
  return nX

# Batch gradient descent
proc BGD(X: Matrix[float64], y: Matrix[float64], n_iter: int, eta: float64): Matrix[float64] =
  var nX = add_ones(normalize(X))
  var (rows, cols) = shape(nX)
  var theta = randomMatrix(1, cols)
  for i in 0..n_iter:
    for c in 0..cols-1:
      var grad = ((nX * theta.t) - y).t * nX.column(c).asMatrix(rows, 1)
      theta[0,c] = theta[0,c] - eta * grad[0, 0]
  return theta.t

# Get the theta for x_train
var theta = BGD(x_train, y_train, 50, 0.01)

# Make predictions for X using theta
proc predict(X: Matrix[float64], theta: Matrix[float64]): Matrix[float64] =
  return add_ones(normalize(X)) * theta

# Round a column
proc round_column(X: Matrix[float64], col: int): Matrix[float64] =
  var (rows, cols) = shape(X)
  var nX = zeros(rows, cols)
  for row in 0..rows-1:
    nX[row, col] = round(X[row, col])
  return nX

# Get predictions and round x_test
var preds = round_column(predict(x_test, theta), 0)

# Accuracy score
proc accuracy(p: Matrix[float64], t: Matrix[float64]): float64 =
  var (rows, cols) = shape(p)
  var c = 0
  for row in 0..rows-1:
    if p[row, 0] == t[row, 0]: c+=1
  return c / rows

echo "Acc: ", accuracy(preds, y_test)