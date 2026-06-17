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

# if the data is exteamly spread out, it is going to drive up the variance to the point predictions become less accurate and useful
# resulting in large residuals.

# Correlation Coefficient = measures the strength of the relationship between 2 variables as a value between -1 and 1.
# closer to 0 = no correlation
# closer to 1 = strong positive, one variable increases, the other proportionally increases.
# closer to -1 = strong negative, one variable increases the other proportionally decreases.

# If there is a strong positive-negative relationship = useful for the model
# if there is NO = just add noise and hurt model accuracy

# pandas corr() 

# 8.3: The correlation coefficient between paire of variables
import pandas as pd

# Read data into Pandas dataframe
df = pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",")

# Print correlations between variables
correlations = df.corr(method='pearson')
print(correlations)
# correlations coefficient 0.957586 = strong positive correlation between the two variables

# more 2 variables = larger grid to compare and the correlation coefficient decreases, indicates a weaker correlation.

# 8.4: Calc. correlation coefficient from scrarch
import pandas as pd
from math import sqrt

points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())
n = len(points)

numerator = n * sum(p.x * p.y for p in points) - \
            sum(p.x for p in points) * sum(p.y for p in points)

denominator = sqrt(n * sum(p.x ** 2 for p in points) - sum(p.x for p in points) ** 2) \
                * sqrt(n * sum(p.y ** 2 for p in points) - sum(p.y for p in points) ** 2)

correlations = numerator / denominator

print(correlations)
# 0.9575860952087218

""" 4. The Statistical Significance """

# The population correlation coefficient with the Greek symbol p (Rho)
# while the sample correlation coefficient is r

# H0:p = 0 (impiles no relationship)
# H1:p =/= 0 (relationship is present, could be positive or negative)

# We use a T-distribution rather than a normal distribution to do hypothesis testing with linear regression *

# 8.5: Calc. the critical value from a T-distribution
from scipy.stats import t

n = 10
lower_cv = t(n-1).ppf(.025)
upper_cv = t(n-1).ppf(.975)

print(lower_cv, upper_cv)
# -2.262157162798206 2.262157162798205
# if test fall ouside this range of -2.262, 2.262 = reject our null hypothesis
# t = test value

# 8.6: Testing significance for linear-looking data
from scipy.stats import t
from math import sqrt

n = 10

lower_cv = t(n-1).ppf(.025)
upper_cv = t(n-1).ppf(.975)

# Correlation coefficient
# Derived from data = https://bit.ly/2KF29Bd
r = 0.957586

# Perform the test
test_value = r / sqrt((1-r**2) / (n-2))

print("TEST VALUE: {}".format(test_value))
print("CRITICAL RANGE: {}, {}".format(lower_cv, upper_cv))

if test_value < lower_cv or test_value > upper_cv:
    print("CORRELATION PROVEN, REJECT HO")

else:
    print("CORRELATION NOT PROVEN, FAILED TO REJECT HO")

# Calc p-value
if test_value > 0:
    p_value = 1.0 - t(n-1).cdf(test_value)
else:
    p_value = t(n-1).cdf(test_value)

# Two-tailed, so multiply by 2
p_value = p_value * 2
print("P-VALUE: {}".format(p_value))

# The test value is around 9.39956 = outside of the range -2.262, 2.262 = reject the null hypothesis
# The correlations is real!

# Having more data will decrease the p-value, if the data gravitates toward a line.
# Low p-value = measuting an engineered and controlled process, not natural


# The more data you have consistently resembles a line = the more significant the p-value for the correlation will be.
# The more scattered or sparse the data, the p-value will increase and indicate the correlation occurred by random chance.

""" 5. Coefficient of Determination"""

# r^2 = Show how much two variables interact with each other.

# 8.7: Creatinga correlation matrix in Pandas
import pandas as pd

df = pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",")

coeff_determination = df.corr(method='pearson') ** 2
print(coeff_determination)
# A coefficient of determination of 0.916971 is 91.69% of the variation in x is explained by y
# the remaining 8.3029% is noise caused by other uncaptured variables

# Correlation does not equal causation, so there could be other variables contributing to the relationship we are looking
# Just because x correlation with y =/= x cauese y. Could be y cause x
# or do not cause each other at all
# Computers have no concept of causality

""" 6. Standard Error of the Estimate """

# SSE, or sum of squared error = measure the overall error of a linear regression
# SSE = Σ(y-y^)^2 

# Se = Σ(y-y^)^2 / n - 2 (the standard error of the estimate)

# y^ or y-hat is each predicted value from the line
# y is represents each actual y-value from the data
# n is the number of the data points

# 8.8: Calc. the standard error of the estimate
import pandas as pd
from math import sqrt

points = list(pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",").itertuples())

n = len(points)

# Regression line
m = 1.939
b = 4.733

# Calculate Standard Error of Estimate
S_e = sqrt((sum((p.y - (m * p.x + b)) ** 2 for p in points)) / (n-2))

print(S_e)
# 1.87406793500129

# A linear regression has 2 variables, not just one,
# we have to increase the uncertainty by one more in our degrees of freedom