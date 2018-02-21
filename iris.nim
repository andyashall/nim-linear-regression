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

proc normalize(X: Matrix[float64], rows: int, cols: int): Matrix[float64] =
  # var nX = zeros(len(X.row(0)), len(X.column(0)))
  var nX = zeros(rows, cols)
  for col in 0..cols-1:
    for row in 0..rows-1:
      nX[row, col] = X[row, col] / max(X.column(col))
  return nX

# proc shape(X: Matrix[float64]): IntArray =
#   var c = 0
#   var r = 0
#   for cols in X.columns

x_train = normalize(x_train, 120, 4)

# echo x_train
var test = matrix(@[
        @[4.0, 6.0, 2.0, 5.0],
        @[3.0, 6.0, 4.0, 1.0],
        @[9.0, 3.0, 5.0, 2.0]])

test = normalize(test, 3, 4)

echo x_train