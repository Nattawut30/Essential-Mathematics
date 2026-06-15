""" Chapter 05: Linear Regression (Contd.) """

# 8.1: Plotting the loss landscape function for linear regression
from sympy import *
from sympy.plotting import plot3d
import pandas as pd

points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())
m, b, i, n = symbols('m b i n')
x, y = symbols('x y', cls=Function)

sum_of_sqaures = Sum((m * x(i) + b - y(i)) ** 2, (i, 0, n)) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)

plot3d(sum_of_sqaures)

""" 1. Overfitting and Variance """

# Our big picture objective is not to minimize the sum of squares but to make accurate predictions on new data.
# This connect-the-dots model is severely overfit, meaning it shaped the regression to the training data too exactly to the point it will predict poorly on new data.
# Overfitting increase variance, predictions are going to be all over the place.

# Overfitting = regression "memorized" the data rather than generalizing itm they are talking about overfitting.
# Bias in the model = prioritize a method as opposed to bending and fitting to exactly what the data says
# Adding bias to a model counteracts overfitting with underfitting or fitting less to the training data

# Ridge regression = adds a further bias to a linear regression in the form of a penalty, fit less to the data.
# Lasso regression = attemp to marginalize noisy variables, useful when remove variables.

""" 2. Stochastic Gradient Descent """

# train on only one sample of dataset on each iteration
# mini-batch gradient descent, multiple samples of the dataset are used (10 or 100 data points) on each iteration.

# Why use only part of the data on each iteration?
# - it reduces computation significantly, as each iteration does not have to travers the entire dataset but only part of it.
# - it reduces overfitting, Exposing the training algorithm to only part of the data on each iteration keeps changing the loss landscape so it does not settle in the loss minimum

# 8.2: Performing stochastic gradient descent for a linear regression
import pandas as pd
import numpy as np

# Input Data
data = pd.read_csv('https://bit.ly/2KF29Bd', header=0)

X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

n = data.shape[0] # rows

# Building the model
m = 0.0
b = 0.0

sample_size = 1  # Sampel size
L = .0001 # The learning Rate
epochs = 1_000_000 # The number of iterations to perform gradient descent

# Performing Stochastic Gradient Descent
for i in range(epochs):
    idx = np.random.choice(n, sample_size, replace=False)
    x_sample = X[idx]
    y_sample = Y[idx]

    # the current predicted value of Y
    Y_pred = m * x_sample + b

    # d/dm derivative of loss function
    D_m = (-2 / sample_size) * sum(x_sample * (y_sample - Y_pred))

    # d/db derivative of loss function
    D_b = (-2 / sample_size) * sum(y_sample - Y_pred)
    m = m - L * D_m # update m
    b = b - L * D_b # update b

    # print progress
    if i % 10000 == 0:
        print(i, m, b)

print("y = {0}x + {1}".format(m, b))
# not gonna converge toward a specific minimum but will end up in a broader neighborhood

# Many algorithm that do approximations are random-based, while some are extreamly useful, others can be sloppy and perform bad

""" 3. The Correlation Coefficient """

