""" Chapter 05: Linear Regression """

# A linear Regression fits a straight line to observed data, attempting to demonstrate
# a linear relationship between variables and make prediction on new data yet to be observed.

# Usuall use split-test-data

# Benefits: It allow us to make predictions on data we have not seen before.

# Another catch is we should not use the linear regression to make prediction outside the range of data we have
# We should not make predictions where x < 0 and x > 10 because we don't have data outside those values.
# Beware the data bias though!

# 7.1: Scikit-Learn to do a linear regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Import Points
df = pd.read_csv('https://bit.ly/3go0Ant', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df[:, :-1].values

# Extract output column (all rows, last column)
Y = df[:, :-1].values

# Fit a line to the points
fit = LinearRegression().fit(X, Y)

# m = 1.7867224, b = -16.51923513
m = fit.coef_.flatten()
b = fit.intercept_.flatten()
print("m = {0}".format(m))
print("b = {0}".format(b))

# Show in Chart
plt.plot(X, Y, 'o') # scatterplot 
plt.plot(X, m*X+b) # line
plt.show()

# 1. Import Data or determinded the data
# 2. Separate the 2 columns into X/Y using pandas
# 3. fit() the LinearRegression model to the input X data and the output Y data
# 4. Then get the m and b coefficients that describe our fitted linear function

""" 1. Residuals and Squared Errors """

# The residual is the numeric difference between the line and the points
# Points above the line = a positive residual, Points below the line = a negative residual
# Error = reflect how wrong our line is in predicting the data

# 7.2: Calculating the residuals for a given line and data
import pandas as pd

# Import points
points = pd.read_csv('https://bit.ly/3go0Ant', delimiter=",").itertuples()

# Test with a given line
m = 1.93939
b = 4.73333

# Calculate the residuals
for p in points:
    y_actual = p.y
    y_predict = m*p.x + b
    residual = y_actual - y_predict
print(residual) # -1.67272, 1.3878900..., -0.5515000..., 2.5091097..., -0.43027999..., -1.3696699...

# Sum of squares = squares each residual or multiplies each residual by itself and sums them
# Absolute values = not working well with calculus deruvatives
# # The sum of squares whould be the sum of all areas where each square has a side length equal to the residual

# 7.3: Calculating the sum of squares fir a given line and data
import pandas as pd

# Import Points
points = pd.read_csv("https://bit.ly/2KF29Bd").itertuples()

# Test with a given line
m = 1.93939
b = 4.73333

sum_of_squares = 0.0

# Calculate sum of sqaures
for p in points:
    y_actual = p.y
    y_predict = m*p.x + b
    residual_squared = (y_predict - y_actual) ** 2
    sum_of_squares += residual_squared

print("Sum of squares = {}".format(sum_of_squares))
# sum of squares = 28.096969704500005

""" 2. Closed Form Equation """
# The heart of training a ML algorithm. share some data and an objective function (the sum of squares)
# and it will find the right coefficients m and b to fullfill that objective.

# Closed Form Equation is use for a simple linear regression with one imput variable.
# 7.4: Calculating m and b for a simple linear regression
import pandas as pd

# Load the data
points = list(pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",").itertuples())

n = len(points)

m = (n*sum(p.x*p.y for p in points) - sum(p.x for p in points) *
sum(p.y for p in points)) / (n*sum(p.x**2 for p in points) -
sum(p.x for p in points)**2)

b = (sum(p.y for p in points) / n) - m * sum(p.x for p in points) / n

print(m, b)
# 1.9393939393994 / 4.73333325
# Calculate m and b are derived from calculus

# Computational Complexity measures how long an algorithm takes as a problem sie grows.

""" 3. Inverse Mstrix Techniques """
# We calculate a vector of coefficients b given a matrix of input variables values X 
# and a vector of output variable values y.

# transposed and inverse operations are performed on the matrix X and combined with matrix multiplication

# 7.5: Inverse and transposed matrices to fit a linear regression
import pandas as pd
from numpy.linalg import inv
import numpy as np

df = pd.read_csv('https://bit.ly/3go0Ant', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1].flatten()

# Add placeholder "1" column to generate intercept
X_1 = np.vstack([X, np.ones(len(X))].T)

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Calculate coefficients for slope and intercept
b = inv(X_1.transpose() @ X_1) @ (X_1.transpose() @ Y)
print(b) # [1.93939394, 4.73333333]

# Predict against the y-values
y_predict = X_1.dot(b)

# since the column is all 1s, it effectively generates the intercept and not just a slope Beta1
# data with a lot of dimensions, computers can start to choke and produce unstable results

# 7.6: QR decomposition to perform a linear regression
import pandas as pd
from numpy.linalg import qr, inv
import numpy as np

df = pd.read_csv('https://bit.ly/3go0Ant', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1].flatter()

# Add placeholder "1" column to generate intercept
X_1 = np.vstack([X, np.ones(len(X))]).transpose()

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Calculate coefficents for slope and intecept
# using QR decomposition
Q, R = qr(X_1)
b = inv(R).dot(Q.transpose()).dot(Y)

print(b) # [1.93939394, 4.733333]
# QR decompostion copes with large amounts of data more easily and is more stable

""" 4. Gradient Descent """
# Optimizartion technique that uses derivatatives and iterations to minimize/maximize a set of parameters against an objective

# We want to minimize our loss and we navigate the loss landscape to do it
# The partial derivative is that. flashlight, allowing us to see the slopes for every parameter
# Calculate the length of this step by taking a fraction of the slope called "learning rate"
# The lower the learning rate, the longer it will take to train and require more iterations.

# A small learning rate will take tiny steps and take an unacceptably long time to get to the bottom but will do so precisely.
# A large learning rate may keep stepping over the minimum to the point he may never reach it no matter how many steps he takes
# A moderate learning rate prob has most balanced step size, having the right trade between speed and accuracy in arriving at the minimum.

# 7.7: Gradient descent to find the minimum of a parabola
import random

def f(x):
    return (x - 3) ** 2 + 4

def dx_f(x):
    return 2*(x - 3)

# The learning rate
L = 0.001

# The number of iterations to perform gradient descent
iterations = 100_000

# Start at a random X
x = random.randint(-15, 15)

for i in range(iterations):

    # Get slope
    d_x = dx_f(x)

    # update x by subtracting the (learning rate) * (slope)
    x -= L * d_x

print(x, f(x)) # 2.999999999, 4.0
# After enough iterations, x will end up at the lowest point of the function or close enough to it where the slope is 0
