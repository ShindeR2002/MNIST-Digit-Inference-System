import matplotlib.pyplot as plt
import numpy as np

def plot_digit_distribution(y_train):
    """Visualizes class balance to ensure no bias toward specific digits."""
    unique, counts = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts, color='skyblue', edgecolor='black')
    plt.xticks(unique)
    plt.xlabel("Digit Class")
    plt.ylabel("Frequency")
    plt.title("MNIST Class Distribution: Ensuring Dataset Balance")
    plt.savefig('images/class_distribution.png')
    plt.show()

def show_sample_grid(X, y):
    """Displays a 5x5 grid of sample digits with their labels."""
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i].reshape(28,28), cmap=plt.cm.binary)
        plt.xlabel(y[i])
    plt.savefig('images/sample_grid.png')
    plt.show()