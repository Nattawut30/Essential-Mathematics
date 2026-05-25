""" Chapter 04: Linear Algebra """
# Very important subject !

# Vector = an arrow in space with a specific direction and length
# The purpose of the vector is to visually represent a piece of data!
# v-> = [x, y]

# 5.1: simple vector in Python
v = [3, 2]
print(v)

# 5.2: Simple Vector in Python using Numpy
import numpy as np
v = np.array([3, 2])
print(v)

# An array of numbers storing data.
# Vectors can have negative directions, and it will impacr when we combine them later.

# 5.3: A 3-Dimensional vector in Python
import numpy as np
v = np.array([4, 1, 2])
print(v)

# 5.4: A 5-Dimensional vector in Python
import numpy as np
v = np.array([6, 1, 5, 8, 3])
print(v)

""" 1. Combining Vectors """
# Combining the movements of 2 vectors into a single vector.
# v-> = [3, 2]
# w-> = [2, -1]
# v-> + w-> = [3 + 2, 2 + (-1)] = [5, 1]

# 5.5: Adding 2-Vectors in Python
from numpy import array

v = array([3, 2])
w = array([2, -1])

v_plus_w = v + w

print(v_plus_w) # [5, 1]
# The point you end at is a new vector.

""" 2. Scaling Vector """
# growing or shrinking a vector's length.
# multiplying or scaling it with a single value
# 2v-> = 2[3, 1] = [3 * 2, 1 * 2] = [6, 2]

# 5.6: Scaling a vector 
from numpy import array

v = array([3, 1])

scaled_v = 2.0 * v

print(scaled_v) # [6, 2]

# Manipulating Data = Manpulating Vectors!

""" 3. Span and Linear Dependence"""
# v-> + w-> = v + w ->
# the whole space of possible vectors is called "span"
# 2 vectors in 2 different directions is called "linearly independent"
# Combining vectos that stuck on the same line, limiting span on the line is called "linearly dependent"

# 2-3 up dimensional usually cause stuck on a plane in a smaller number
# * A lot of problems is more difficult or unsolvable when they are linearly dependent *

""" 4. Linear Transformations """
# Control directions
# Use a vector to transform another vector in a function-like manner.
# i^ and i^ (i-hat and j-hat)

# v^ = i^ + j^
# 3i^ = 3[1, 0] = [3, 0]
# 2j^ = 2[0, 1] = [0, 2]
# v^ = [3, 2]

""" 5. Matrix """
# i^ is the first column [a, c] and j^ is the column [b, d]
# [xnew, ynew] = [ax + by, cx + dy]

# 5.7: Matrix vector multiplication in NumPy

from numpy import array

# Compose basis matrix with i-hat and j-hat

basis = array([[3, 0], [0, 2]])

# Declare vector v
v = array([1, 1])

# Make a new vector
# tranforming v with the dot product
new_v = basis.dot(v)

print(new_v) # [3, 2]

# transpose = swap the columns and rows
# Numpy's array() function will do the opposite orientation we want
 
# 5.8: Separating the basis vectors and applying them as a tranformation

from numpy import array

# Declare i-hat and j-hat

i_hat = array([2, 0])
j_hat = array([0, 3])

# Compose basis matrix using i-hat and j-hat
# transpose rows into columns
basis = array([i_hat, j_hat]).transpose()

v = array([1, 1])

new_v = basis.dot(v)

print(new_v) # [2, 3]

# if v^ = [2, 1] and i^ and j^ is [1, 0] and [0, 1]
# tranform fast to i^ = [2, 0] and j^ = [0, 3], v^ = [4, 3]

# 5.9: Transforming a vector fast

from numpy import array

i_hat = ([2, 0])
j_hat = ([0, 3])

basis = array([i_hat, j_hat]).transpose()

v = array([2, 1])

new_v = basis.dot(v)

print(new_v)

# 5.10: A complicated transformation

from numpy import array

i_hat = ([2, 3])
j_hat = ([2, -1]) # you can see it -1

basis = array([i_hat, j_hat]).transpose()

v = array([2, 1])

new_v = basis.dot(v)

print(new_v) # [6, 5]
# sheared, rotated, and flipped space.

# 3-dimensional matrix = i^, j^, and k^

""" 6. Matrix Multiplication """
# Applying multiple transformation to a vector space

# Over-and-Down! method
# [a, b] [e f] = [ae + bg + af + bh]
# [c, d] [g h] = [ce + dg + cf + dh]

# Numpy = use matmul() or @ operator

# 5.11: Combining two transformations
from numpy import array

# Transformation 1
i_hat1 = array([0, 1])
j_hat2 = array([-1, 0])
transform1 = array([i_hat1, j_hat2]).transpose()

# Transformation 2
i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()

# Combine Transformation 1 & 2
combined = transform2 @ transform1

# Test
print("Combined Matrix: \n {}".format(combined))

v = array([1, 2])
print(combined.dot(v)) # [-1, 1]
# The order you apply each transformation matters!
# T2 on T1 only

# 5.12: Reverse applying the transformation
from numpy import array

# T1
i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])

# T2
i_hat2 = array([1, 0])
j_hat2 = array([1, 1])

# Combine transformation, but in reverse this time
combined = transform1 @ transform2 # reverse

print("Combined Matrix: \n {}".format(combined))

v = array([1, 2])
print(combined.dot(v)) # [-2, 3]
# Wrong orders, different answer!