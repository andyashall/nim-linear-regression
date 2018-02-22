import sknim

var train = read_csv("./data/boston.csv", 506, 14)

var x = drop_column(train, 4)
var y = get_column(train, 4)

var (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.5, 92)

var theta = MBGD(x_train, y_train, 500, 0.01, 64)

var preds = predict(x_test, theta)

# proc stdev(X: Matrix[float64]): float64 =
#   var (rows, cols) = shape(X)
#   return sqrt(average(X) / float(rows))

echo "Acc: ", explained_variance_score(preds, y_test)