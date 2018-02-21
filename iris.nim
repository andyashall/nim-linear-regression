import neo
import strutils
from random import random, randomize
import times, os, parsecsv, streams, math

# Reads csv file to matrix ("filepath", n_rows, n_cols) : how can it detect n_rows and n_cols?
proc read_csv(fn: string, rows: int, cols: int): (Matrix[float64], Matrix[float64]) =
  var x = zeros(rows, cols-1, float64)
  var y = zeros(rows, 1, float64)
  var s = newFileStream(fn, fmRead)
  if s == nil: quit("cannot open the file" & paramStr(1))
  var csv: CsvParser
  open(csv, s, fn)
  var row = 0
  var col = 0
  while readRow(csv):
    col = 0
    for val in items(csv.row):
      if col <= cols-2:
        x[row, col] = parseFloat(val)
      else:
        y[row, 0] = parseFloat(val)
      col += 1
    row += 1
  close(csv)
  return (x, y)

var (x_train, y_train) = read_csv("./data/iris_training.csv", 120, 5)

var (x_test, y_test) = read_csv("./data/iris_test.csv", 30, 5)

# Get the shape of a matrix
proc shape(X: Matrix[float64]): (int, int) =
  var c = 0
  var r = 0
  for val in X.column(0):
    r+=1
  for val in X.row(0):
    c+=1
  return (r, c)

# Normalize matrix columns
proc normalize(X: Matrix[float64]): Matrix[float64] =
  var (rows, cols) = shape(X)
  var nX = zeros(rows, cols)
  for col in 0..cols-1:
    for row in 0..rows-1:
      nX[row, col] = X[row, col] / max(X.column(col))
  return nX

# Adds a column of 1s to a matrix
proc addOnes(X: Matrix[float64]): Matrix[float64] =
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
  var nX = addOnes(normalize(X))
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
  return addOnes(normalize(X)) * theta

# Round a column
proc roundCol(X: Matrix[float64], col: int): Matrix[float64] =
  var (rows, cols) = shape(X)
  var nX = zeros(rows, cols)
  for row in 0..rows-1:
    nX[row, col] = round(X[row, col])
  return nX

# Get predictions and round x_test
var preds = roundCol(predict(x_test, theta), 0)

# Accuracy score
proc accuracy(p: Matrix[float64], t: Matrix[float64]): float64 =
  var (rows, cols) = shape(p)
  var c = 0
  for row in 0..rows-1:
    if p[row, 0] == t[row, 0]: c+=1
  return c / rows

echo "Acc: ", accuracy(preds, y_test)