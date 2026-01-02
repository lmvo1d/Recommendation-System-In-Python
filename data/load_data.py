import numpy as np

def load_movielens(path="data/ml-100k/u.data"):
    data = np.loadtxt(path)

    user_ids = data[:, 0].astype(int)
    item_ids = data[:, 1].astype(int)
    ratings = data[:, 2]

    num_users = user_ids.max()
    num_items = item_ids.max()

    R = np.zeros((num_users, num_items))

    for u, i, r in zip(user_ids, item_ids, ratings):
        R[u - 1, i - 1] = r

    return R
