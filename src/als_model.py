import numpy as np

def als(ratings, k=2, lambda_=0.1, iterations=20):
    num_users, num_items = ratings.shape

    U = np.random.rand(num_users, k)
    V = np.random.rand(num_items, k)

    for _ in range(iterations):
        for u in range(num_users):
            V_u = V[ratings[u] > 0]
            R_u = ratings[u][ratings[u] > 0]

            if len(R_u) == 0:
                continue

            A = V_u.T @ V_u + lambda_ * np.eye(k)
            b = V_u.T @ R_u
            U[u] = np.linalg.solve(A, b)

        for i in range(num_items):
            U_i = U[ratings[:, i] > 0]
            R_i = ratings[:, i][ratings[:, i] > 0]

            if len(R_i) == 0:
                continue

            A = U_i.T @ U_i + lambda_ * np.eye(k)
            b = U_i.T @ R_i
            V[i] = np.linalg.solve(A, b)

    return U, V

def predict(U, V):
    return U @ V.T
