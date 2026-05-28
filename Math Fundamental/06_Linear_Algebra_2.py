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

# Extract the coefiiciants into matrix A, the values on the right side of the equation into matrix B
# AX = B
 # 6.5: The inverse and identity matrix
from sympy import *

# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72

A = Matrix([
    [4, 2, 4],
    [5, 3, 7],
    [9, 3, 6],
])

# dot product between A and its inverse
# will product identity function
inverse = A.inv()
identity = inverse * A

print("INVERSE: {}".format(inverse))

print("IDENTITY: {}".format(identity))

# The lack of floating point precision will not affect our answers too badly,
# 6.6: Numpy to solve a system of equations
from numpy import array
from numpy.linalg import inv

# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72

A = array([
    [4, 2, 4],
    [5, 3, 7],
    [9, 3, 6],
])

B = array([
    44,
    56,
    72
])

X = inv(A).dot(B)

print(X) # [2. 34. -8]
# So we knew it x = 2, y = 34, z = -8

# 6.7: The full solution in SymPy to solve a system of equations
from sympy import *

# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72

A = Matrix([
    [4, 2, 4],
    [5, 3, 7],
    [9, 3, 6],
])

B = Matrix([
    44,
    56,
    72
])

X = A.inv() * B

print(X) # Matrix([[2], [34], [-8]])

""" 4. Eigenvectors and Eigenvalues """

# Matrix decompostion = breaking up a matrix into its basic components much like a factoring numbers
# Useful for tasks like finding inverse matrices and calculating determinants

# eigendecomposition is often used for ml and principle compenent analysis
# helpful for breaking up a matrix into components that are easier to work with in different ml tasks
# however, not all matrices can be decomposed into an eigenvector and eigenvalue

# 6.8: Performing eigendecomposition in NumPy
from numpy import array, diag
from numpy.linalg import eig, inv

A = array([
    [1, 2],
    [4, 5],
])

eigenvals, eigenvecs = eig(A)

print("EIGENVALUES")
print(eigenvals)
print("\nEIGENVECTORS")
print(eigenvecs)

# reconstruct A
# Diagonal form means the vector is padded into a matrix of zeroes and occupies the diagonal line in a similar
# pattern to an identity matrix

# 6.9: Decomposing and recomposing a matrix in NumPy
from numpy import array, diag
from numpy.linalg import eig, inv

A = array([
    [1, 2],
    [4, 5]
])

eigenvals, eigenvecs = eig(A)

print("EIGENVALUES")
print(eigenvals)
print("\nEIGENVECTORS")
print(eigenvecs)

print("\nREBUILD MATRIX")
Q = eigenvecs
R = inv(Q)

L = diag(eigenvals)
B = Q @ L @ R

print(B)

# As you can see the matrix we rebuild is the one we started with.

