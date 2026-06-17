""" Chapter 05: Linear Regression (Contd.) """

""" 1. Prediction Intervals """

# Our regression is only as good as our sample
# Our linear regression line also has a normal distribution running along it.
# the "mean" is shifting along the line.

# A regression line serves as the shifting "mean" of our bell curve,
# and the spread of the data around the line reflects the variance/standard deviation

# There is a confidence interval around each y prediction, and this is known as "a prediction interval"
# We need to get the margin of error and plus/minus it around the predicted y-value
# A critical value from the T-distribution as well as the standard error of the estimate.

# 9.1: Calc. a prediction interval of vet visit for a dog that's 8.5 years old
import pandas as pd
from scipy.stats import t
from math import sqrt

points = list(pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",").itertuples())

n = len(points)

# Linear Regression Line
m = 1.939
b = 4.733

# Calculate Prediction Interval for x = 8.5
x_0 = 8.5
x_mean = sum(p.x for p in points) / len(points)

t_value = t(n-2).ppf(.975)

standard_error = sqrt(sum((p.y - (m * p.x + b)) ** 2 for p in points) / (n-2))

margin_of_error = t_value * standard_error * \
                sqrt(1 + (1 / n) + (n * (x_0 - x_mean) ** 2) / \
                     (n * sum(p.x ** 2 for p in points) - \
                      sum(p.x for p in points) ** 2))

predicted_y = m * x_0 + b

# Calculate prediction interval
print(predicted_y - margin_of_error, predicted_y + margin_of_error)
# 16.46251687560351 25.966483124396493

# There's a 95% probability an 8.5 year old dog will visit the vet between 16.46 and 25.96 times
# It also captures a range rather than a single value, and thus accounts for uncertainty.

""" 2. Train/Test Splits """

