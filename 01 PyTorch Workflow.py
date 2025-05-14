import torch
from torch import nn
import matplotlib.pyplot as plt

print(torch.__version__)

# Data preparing and loading
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)  # (50, 1)
y = weight * X + bias
print(X[:10], y[:10])

# Split data into sets - training(60-80%), validation(10-20%), testing(10-20%)

# 80% training, 20% testing
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(y_train), len(X_test), len(y_test))


def plot_predictions(train_data=X_train, 
                    train_labels=y_train,
                    test_data=X_test,
                    test_labels=y_test,
                    predictions=None):
    """Plots training data, test 
    data and compares predictions"""
    plt.figure(figsize=(10, 7))

    #  Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    #  Plot testing data in purple
    plt.scatter(test_data, test_labels, c="purple", s=4, label="Testing data")  

    #  Are there any predictions?
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})


plot_predictions()