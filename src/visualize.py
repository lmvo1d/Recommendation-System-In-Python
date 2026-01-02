import matplotlib.pyplot as plt

def plot_labeled_matrix(matrix, title):
    plt.imshow(matrix, cmap="hot")
    plt.colorbar(label="Predicted Rating")

    plt.xticks(range(matrix.shape[1]),
               [f"Item {i+1}" for i in range(matrix.shape[1])])
    plt.yticks(range(matrix.shape[0]),
               [f"User {i+1}" for i in range(matrix.shape[0])])

    plt.title(title)
    plt.xlabel("Items")
    plt.ylabel("Users")
    plt.show()
