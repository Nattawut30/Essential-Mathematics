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

# They are dealing with so much data they do not have the time or technical ability to do so
# When operating with massive amounts of data and variables you cannot sift through all of that
# Scikit-learn does not support confidence intervals and P-values, as these two techniques are open problems for higher-dimensional

# Train/test split, which typically 1/3 of the data is set aside for testing
# the other 2/3 is used for traning.

# The training dataset is used to fit the linear regression
# The testing dataset is used to measure the linear regression's performance on data

# Remember: fitting a regression is synnonymous with "training" The latter word is used by machine learning practitioners.

# 9.2: Doing a train/test split on linear regression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('https://bit.ly/3cIH97A', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Separate training and testing data
# This leaves a third of the data out for testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3)

model = LinearRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("r^2: %.3f" % result)

# train_test_split() will take dataset (X and Y column), shuffle it, and return our training and testing datasets
# based on our testing-dataset size.
# LinearRegression'sfit() use it to fit on the training datasets X_train and Y_train
# score() function on the testing datasets X_test and Y_test to evaluate the r^2
# The higher the r^2 is for our testing dataset, the better

# Alternative way to testing dataset across each 1/3 fold = cross-validation
# This is the gold standard of validation techniqures.

# 9.3: Using three-fold cross-validation for a linear regression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv('https://bit.ly/3cIH97A', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Perform a simple linear regression
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(results)
print("MSE: mean=%.3f (stdev-%.3f)" % (results.mean(), results.std()))
# In this case the MSE is averaged alongside its standard deviation to show how consistently each test performed.

# random-fold validation = repeatly shuffle and train/test splt your data an unlimited number of times

# 9.4: Using a random-fold validation for a linear regression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit

df = pd.read_csv('https://bit.ly/38XwbeB', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Perform a simple linear regression
kfold = ShuffleSplit(n_splits=10, test_size=.33, random_state=7)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)

print(result)
print("mean=%.3f (stdev-%.3f)" % (results.mean(), results.std()))

# A train/test split is going to provide a way to measure how well you linear regression will perform on data it has not seen before.

# Train/Test splits are not guarantees!
# holding out another dataset or "validation set" is sometimes necessary
# You can use the validation dataset as one last stopgap to see if p-hacking caused you to overfit to your testing dataset.
# sometimes we got it all including training, testing, and validation but it still biased to begin anyway

""" 3. Multiple Linear Regression """

# r^2 is fine but the standard error and gets harder with more than 2 vriables

# 9.4: A linear regression with two input variables
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('https://bit.ly/2X1HWH7', delimiter=",")

# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]

# Extract output column (all rows, last column)
Y = df.values[:, -1]

# Training
fit = LinearRegression().fit(X,Y)

# print coefficients
print("Coefficients = {0}".format(fit.coef_))
print("Intercept = {0}".format(fit.intercept_))
print("z = {0} + {1}x + {2}y".format(fit.intercept_, fit.coef_[0], fit.coef_ [1]))

# When a model becomes so inundated with variables it starts to lose explainbility
# The machine learning start to come in and treat the model as a black box

# Analyze the relationships between each pair of variables using a correlation matrix,
# It will help your efforts to create a productive machine learning model