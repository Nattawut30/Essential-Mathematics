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

