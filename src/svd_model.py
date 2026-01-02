import numpy as np
from scipy.linalg import svd

def compute_svd(ratings, k=2):
    U, sigma, Vt = svd(ratings)

    U_k = U[:, :k]
    sigma_k = np.diag(sigma[:k])
    Vt_k = Vt[:k, :]

    reconstructed = U_k @ sigma_k @ Vt_k
    return reconstructed
