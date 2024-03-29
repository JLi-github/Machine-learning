#!/usr/bin/env python3

from statistics import mean
import numpy as np 
import random
import matplotlib.pyplot as plt

#xs = np.array([1,2,3,4,5,6,7], dtype = np.float64)
#ys = np.array([3,4,5,6,7,8,7], dtype = np.float64)
def create_dataset(hm, variance, step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val+=step
        elif correlaltion and correlatlion == "neg":
            val -=step
    xs = [i for i in range(len(ys))]
    print("xs:", xs)
    print("ys:", ys)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

xs, ys = create_dataset(40,40,2, correlation = "pos")
plt.scatter(xs,ys)

plt.show()

def best_fit_slope_and_intercept (xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) ** 2) - mean(xs ** 2))
    b = mean(ys)- m*mean(xs)
    return m,b

def squared_error(y_orig, y_line):
    return sum((y_line - y_orig) ** 2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

m,b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = m*predict_x+b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()










