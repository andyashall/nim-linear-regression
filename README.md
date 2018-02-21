## Todo

- [ ] Detect CSV n rows and columns
- [ ] And remove column header
- [x] Is there a way to drop a column from matrix?

```nim
var y = x.column(11)

proc dropCol(X: Matrix[float64], drop: int): Matrix[float64] =
  var (rows, cols) = shape(X)
  var nX = zeros(rows, cols-1)
  var newCol = 0
  for col in 0..cols-1:
    if col != drop:
      for row in 0..rows-1:
        nX[row, newCol] = X[row, col]
      newCol+=1
  return nX

x = dropCol(x, 11)

```

- [x] Add a train test split function
- [ ] Optimize and clean up code