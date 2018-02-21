import neo
import strutils
from random import random, randomize
import times, os, parsecsv, streams

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

proc shape(X: Matrix[float64]): (int, int) =
  var c = 0
  var r = 0
  for val in X.column(0):
    r+=1
  for val in X.row(0):
    c+=1
  return (r, c)

proc normalize(X: Matrix[float64]): Matrix[float64] =
  var (rows, cols) = shape(X)
  var nX = zeros(rows, cols)
  for col in 0..cols-1:
    for row in 0..rows-1:
      nX[row, col] = X[row, col] / max(X.column(col))
  return nX

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


# echo x_train
# var test = matrix(@[
#         @[4.0, 6.0, 2.0, 5.0],
#         @[3.0, 6.0, 4.0, 1.0],
#         @[9.0, 3.0, 5.0, 2.0]])

# test = normalize(test)

# echo test

# echo addOnes(normalize(test))

proc BGD(X: Matrix[float64], y: Matrix[float64], n_iter: int, eta: float64): Matrix[float64] =
  var nX = addOnes(normalize(X))
  var (rows, cols) = shape(nX)
  var theta = randomMatrix(1, cols)
  for i in 0..n_iter:
    for c in 0..cols-1:
      var grad = ((nX * theta.t) - y).t * nX.column(c).asMatrix(rows, 1)
      theta[0,c] = theta[0,c] - eta * grad[0, 0]
  return theta.t

var theta = BGD(x_train, y_train, 5, 0.01)

proc predict(X: Matrix[float64], theta: Matrix[float64]): Matrix[float64] =
  return addOnes(normalize(X)) * theta

var preds = predict(x_test, theta)

