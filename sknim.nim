import neo
import strutils
from random import random, randomize, shuffle
import os, parsecsv, streams, math

# Reads csv file to matrix ("filepath", n_rows, n_cols) : how can it detect n_rows and n_cols?
proc read_csv(fn: string, rows: int, cols: int): Matrix[float] =
  var
    x = zeros(rows, cols, float)
    s = newFileStream(fn, fmRead)
  if s == nil: quit("cannot open the file" & paramStr(1))
  var csv: CsvParser
  open(csv, s, fn)
  var
    row = 0
    col = 0
  while readRow(csv):
    col = 0
    for val in items(csv.row):
      x[row, col] = parseFloat(val)
      col += 1
    row += 1
  close(csv)
  return x

# Get the shape of a matrix
proc shape(X: Matrix[float]): (int, int) =
  var 
    c = 0
    r = 0
  for val in X.column(0):
    r+=1
  for val in X.row(0):
    c+=1
  return (r, c)

# Creates a random seq of indicies for n length
proc permutation(n: int, seed: int): seq[int] =
  randomize(seed)
  var arr = newSeq[int](n)
  for i in 0..n-1: arr[i] = i
  shuffle(arr)
  return arr

# Splits X and y into training and testing data : ts is test set size between 0 - 1, seed is seed for randomize
proc train_test_split(X: Matrix[float], y: Matrix[float], ts: float, seed: int): (Matrix[float], Matrix[float], Matrix[float], Matrix[float]) =

  var
    (rows, cols) = shape(X)
    ind = permutation(rows, seed)

  var
     train_size = int(round((float(rows) * (1-ts)) + 0.4))
     test_size = int(round((float(rows) * ts) - 0.1))
     x_train = zeros(train_size, cols)
     x_test = zeros(test_size, cols)
     y_train = zeros(train_size, 1)
     y_test = zeros(test_size, 1)
     count = 0

  for col in 0..cols-1:
    count = 0
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

  # Drop a column from matrix
proc drop_column(X: Matrix[float], drop: int): Matrix[float] =
  var
    (rows, cols) = shape(X)
    nX = zeros(rows, cols-1)
    newCol = 0
  for col in 0..cols-1:
    if col != drop:
      for row in 0..rows-1:
        nX[row, newCol] = X[row, col]
      newCol+=1
  return nX

# Return column n from matrix as (r, 1) matrix
proc get_column(X: Matrix[float], get: int): Matrix[float] =
  var (rows, cols) = shape(X)
  return X.column(get).asMatrix(rows, 1)

# Normalize matrix columns
proc normalize(X: Matrix[float]): Matrix[float] =
  var
    (rows, cols) = shape(X)
    nX = zeros(rows, cols)
  for col in 0..cols-1:
    for row in 0..rows-1:
      nX[row, col] = X[row, col] / max(X.column(col))
  return nX

# Adds a column of 1s to a matrix
proc add_ones(X: Matrix[float]): Matrix[float] =
  var
    (rows, cols) = shape(X)
    nX = zeros(rows, cols+1)
  for col in 0..cols:
    if col == cols:
      for row in 0..rows-1:
        nX[row, col] = 1 
    else:   
      for row in 0..rows-1:
        nX[row, col] = X[row, col]
  return nX

# Batch gradient descent
proc BGD(X: Matrix[float], y: Matrix[float], n_iter: int, eta: float): Matrix[float] =
  var
    nX = add_ones(normalize(X))
    (rows, cols) = shape(nX)
    theta = randomMatrix(1, cols)
  for i in 0..n_iter:
    for c in 0..cols-1:
      var grad = ((nX * theta.t) - y).t * nX[0..rows-1, c..c]
      theta[0, c] = theta[0, c] - eta * grad[0, 0]
  return theta.t

# Stochastic gradient descent
proc SGD(X: Matrix[float], y: Matrix[float], n_epoch: int, eta: float): Matrix[float] =
  var
    nX = add_ones(normalize(X))
    (rows, cols) = shape(nX)
    theta = randomMatrix(1, cols)
  for epoch in 0..n_epoch:
    for r in 0..rows-1:
      for c in 0..cols-1:
        randomize()
        var
          i = random(rows-1)
          xi = nX[i..i, 0..cols-1]
          yi = y[i..i, 0..0]
          grad = ((xi * theta.t) - yi).t * xi[0..0, c..c]
        theta[0, c] = theta[0, c] - eta * grad[0, 0]
  return theta.t

# Mini-Batch gradient descent
proc MBGD(X: Matrix[float], y: Matrix[float], n_epoch: int, eta: float, batch_size: int): Matrix[float] =
  var
    nX = add_ones(normalize(X))
    (rows, cols) = shape(nX)
    theta = randomMatrix(1, cols)
  for epoch in 0..n_epoch:
    for i in countup(0, rows-batch_size, batch_size):
      for c in 0..cols-1:
        var
          ind = permutation(rows-1, 29)
          xi = nX[i..i+batch_size-1, 0..cols-1]
          yi = y[i..i+batch_size-1, 0..0]
          grad = ((xi * theta.t) - yi).t * xi[0..batch_size-1, c..c]
        theta[0, c] = theta[0, c] - eta * grad[0, 0]
  return theta.t

# Make predictions for X using theta
proc predict(X: Matrix[float], theta: Matrix[float]): Matrix[float] =
  return add_ones(normalize(X)) * theta

# Round a column
proc round_column(X: Matrix[float], col: int): Matrix[float] =
  var
    (rows, cols) = shape(X)
    nX = zeros(rows, cols)
  for row in 0..rows-1:
    nX[row, col] = round(X[row, col])
  return nX

# Accuracy score
proc accuracy_score(p: Matrix[float], t: Matrix[float]): float =
  var
    (rows, cols) = shape(p)
    c = 0
  for row in 0..rows-1:
    if p[row, 0] == t[row, 0]: c+=1
  return c / rows

proc average(y: Matrix[float]): float =
  var 
    (rows, cols) = shape(y)
    sum = 0.0
  for i in 0..rows-1: sum+=y[i,0]
  return sum / float(rows)

# explained_variance_score
proc explained_variance_score(p: Matrix[float], t: Matrix[float]): float =
  var
    diff_avg = average(t - p)
    true_avg = average(t)
    numer = t - p
    denom = t
    c = 0
  for i in numer:
    numer[c, 0] = pow(i - diff_avg, 2)
    denom[c, 0] = pow(denom[c, 0] - true_avg, 2)
    c+=1
  var
    numerator = average(numer)
    demoni = average(denom)
  return 1 - (numerator / demoni)

export read_csv, shape, permutation, train_test_split, drop_column, get_column, normalize, add_ones, BGD, SGD, MBGD, predict, round_column, accuracy_score, average, explained_variance_score