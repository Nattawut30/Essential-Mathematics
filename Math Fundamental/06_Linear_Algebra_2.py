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
