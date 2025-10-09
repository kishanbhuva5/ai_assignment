import numpy as np

A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

A_inverse = np.linalg.inv(A)

prod1 = A @ A_inverse
prod2 = A_inverse @ A

np.set_printoptions(precision=6, suppress=True)
print("Matrix A:")
print(A)
print("\nInverse of A:")
print(A_inverse)
print("\nA @ A_inverse:")
print(prod1)
print("\nA_inverse @ A:")
print(prod2)

#finally checking the products are close to identity matrix
identity = np.eye(3)
print("\nIs A @ A_inv close to identity? ->", np.allclose(prod1, identity))
print("Is A_inv @ A close to identity? ->", np.allclose(prod2, identity))
