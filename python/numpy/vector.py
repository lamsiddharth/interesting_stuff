import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

vector_addition = a + b
print("Vector Addition:", vector_addition)


scalar_multiplication = 2 * a
print("Scalar Multiplication:", scalar_multiplication)

dot_product = np.dot(a, b)
print("Dot Product:", dot_product)

v = np.array([1, 2, 3])
m = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])
dot_product_matrix = np.dot(v, m)
print("Dot Product with Matrix:", dot_product_matrix)