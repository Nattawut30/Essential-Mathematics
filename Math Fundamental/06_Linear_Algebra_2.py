""" Chapter 04: Linear Algebra (contd.) """

""" 1. Determinants """
# Linear Transformation "expand" or "squish", it make some space
# A determinant = measures how a linear transformation scales an area

# 6.1: Calc. a determinant
from numpy.linalg import det
from numpy import array

i_hat = array([3, 0])
j_hat = array([0, 2])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)

print(determinant) # 6.0 blocks 

# BTW, a shears and rotations should not affect the determinant, as the area will not change!
# 6.2: A determinant for a shear
from numpy.linalg import det
from numpy import array

i_hat = array([1, 0])
j_hat = array([1, 1])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)

print(determinant) # 1.0 block

# Scaling will increase or decrease the determinants as that will increase/decrease the sampled area
# orientation flip = i^ and j^ swap clockwise positions = the determinant will be negative!

# 6.3: A negative Determinant
from numpy.linalg import det
from numpy import array

i_hat = array([-2, 1])
j_hat = array([1, 2])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)

print(determinant) # -5.0

# if you have a determinant of 0, that means all of the space has been squished into a lesser dimension.
# the area and volume respectively in both cases are 0!

# 6.4: A determinant of zero
from numpy.linalg import det
from numpy import array

i_hat = array([-2, 1])
j_hat = array([3, -1.5])

basis = array([i_hat, j_hat]).transpose()

determinant = det(basis)

print(determinant) # 0.0
# So testing for a 0 determinant is highly helpful to determine if a transformation has linear dependence.

""" 2. Special Types of Matrices """

# Sqaure Matrix
# a matrix that has an equal number of rows and columns
# primarily used to represent linear transformations

# Identity Matrix
# a square matrix that has a diagonal of 1s while the other values are 0

# Inverse Matrix
# a matrix that undoes the transformation of another matrix
# The inverse of matrix A is called A^-1

# Diagonal Matrix
# non-zero values while the rest of the values are 0

# Triangular Matrix
# A diagonal of non-zero values in front of a triangle of values while the rest of the values are 0

# Sparse Matrix
# The matrix that are mostly zero and have very few non-zero

""" 3. Systems of Equations and Inverse Matrices """

