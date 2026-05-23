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