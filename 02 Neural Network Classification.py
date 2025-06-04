#Classification is a problem of predicting whether something is one thing or another

import torch
from torch import nn
# import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        #ceate 2 nn.Linear layers
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        return self.layer_2(self.layer_1(x))


class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        # create 3 layers
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(x)))


class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=20)
        self.layer_2 = nn.Linear(in_features=20, out_features=20)
        self.layer_3 = nn.Linear(in_features=20, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


n_samples = 1000

# create circles
X, y = make_circles(n_samples, 
                    noise=0.03,
                    random_state=42)

print(X[:5], y[:5])
print(X.shape, y.shape)


circles = pd.DataFrame({"X0": X[:, 0],
                        "X1": X[:, 1],
                        "label": y})
print(circles.head(10))

# visualize the data

plt.scatter(x=circles["X0"], 
            y=circles["X1"],
            c=circles["label"],
            cmap="winter")
# plt.show()

x_sample = X[0]
y_sample = y[0]

print(f"X: {x_sample}, y: {y_sample}")
print(f"X shape: {x_sample.shape}, y shape: {y_sample.shape}")

# turn the data into tensors
X_tensor = torch.from_numpy(X).type(torch.float)
y_tensor = torch.from_numpy(y).type(torch.float)

print(f"X tensor: {X_tensor[:5]}, y tensor: {y_tensor[:5]}")
print(f"X tensor shape: {X_tensor.shape}, y tensor shape: {y_tensor.shape}")

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, 
                                                    y_tensor, 
                                                    test_size=0.2,
                                                    random_state=42)
print("len(X_train), len(y_train), len(X_test), len(y_test)", len(X_train), len(y_train), len(X_test), len(y_test))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# create model
model_0 = CircleModelV0().to(device)

print("Model 0: ", model_0)
print("Model 0 state dict: ", model_0.state_dict())

# another method to create a model
model_1 = torch.nn.Sequential(
    torch.nn.Linear(in_features=2, out_features=5),
    torch.nn.Linear(in_features=5, out_features=1)
).to(device)

print("Model 1: ", model_1)
print("Model 1 state dict: ", model_1.state_dict())

# check untrained predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
    # print("Untrained predictions length: ", len(untrained_preds))
    # print("Untrained predictions shape: ", untrained_preds.shape)
    # print("Untrained predictions: ", untrained_preds[:10])
    # print("Expected output: ", y_test[:10])

# pick loss function and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                              lr=0.1)  # Stochastic Gradient Descent

# calculate accuracy - out of 100 examples, how many are correct
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct / len(y_pred) * 100
    return acc

# building a training (and a testing loop) loop in pyTorch

# going from raw logits -> prediction probabilities -> prediction labels

y_logits = model_0(X_test.to(device))
print("Logits: ", y_logits[:10])

# use sigmoid activation function on our model to get prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
pred_labels = torch.round(y_pred_probs[:10])

# training loop
epochs = 100

for epoch in range(epochs):
    # training
    model_0.train()
    # forward pass
    y_logits = model_0(X_train.to(device))
    y_pred = torch.round(torch.sigmoid(y_logits))

    # calculate loss
    loss = loss_fn(y_logits.squeeze(), # nn.BCEWithLogitsLoss() expects raw logits
                   y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    # optimizer zero grad
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # optimizer step
    optimizer.step()

    ### testing
    model_0.eval()
    with torch.inference_mode():
        # forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # calculate loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                                y_pred=test_pred)
        
        # print out what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | "
                  f"Train loss: {loss:.5f} | "
                  f"Train acc: {acc:.2f}% | "
                  f"Test loss: {test_loss:.5f} | "
                  f"Test acc: {test_acc:.2f}%")
            
import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download.")
else:
    print("Downloading helper_functions.py...")
    url = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
    response = requests.get(url)
    with open("helper_functions.py", "wb") as f:
        f.write(response.content)
    print("Download complete.")

from helper_functions import plot_predictions, plot_decision_boundary

# plot predictions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Training")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
# plt.show()

model_1 = CircleModelV1().to(device)
print("Model 1: ", model_1.state_dict())

# create a loss function
loss_fn = nn.BCEWithLogitsLoss()

# create an optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

torch.manual_seed(42)

epochs = 1000

# Put data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    ### training
    model_1.train()
    # forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    # calculate loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    # optimizer zero grad
    optimizer.zero_grad()
    # backward pass
    loss.backward()
    # optimizer step
    optimizer.step()

    ### test
    model_1.eval()
    with torch.inference_mode():
        # forward pass
        test_logits = model_1(X_test).squeeze()
        pred_labels = torch.round(torch.sigmoid(test_logits))
        # calculate the loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=pred_labels)
        # print out what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}," 
                  f"Acc: {acc:.2f}% | Test loss: {test_loss:.5f},"
                  f" Test acc: {test_acc:.2f}%")
            
# create data 
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias 

# check the data
print(len(X_regression))
print(X_regression[:5], y_regression[:5])

# create train and test splits
train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

print(f"Lengths:\nX_train_regression: {len(X_train_regression)}, "
      f"y_train_regression: {len(y_train_regression)}, "
      f"X_test_regression: {len(X_test_regression)}, "
      f"y_test_regression: {len(y_test_regression)}")

# # visualize the data
# plt.figure(figsize=(12, 6))
# plt.scatter(X_train_regression, y_train_regression, c="b", label="Train data")
# plt.scatter(X_test_regression, y_test_regression, c="green", label="Test data")
# plt.title("Regression Data")
# plt.xlabel("X")
# plt.ylabel("y")
# plt.show()

model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.01)

torch.manual_seed(42)

epochs = 1000

X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

#  training loop 

for epoch in range(epochs):
    ### training
    model_2.train()
    # forward pass
    y_pred = model_2(X_train_regression)
    # calculate the loss
    loss = loss_fn(y_pred, y_train_regression)
    # optimizer zero grad
    optimizer.zero_grad()
    # loss backwards
    loss.backward()
    # optimizer step
    optimizer.step()

    ### let's test
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regression)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | "
                  f"Train loss: {loss:.5f} |"
                  f"Test loss: {test_loss:.5f}")
    
model_2.eval()
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression,
                 predictions=y_preds)
# plt.show()

### final circle model cration

model_3 = CircleModelV2().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(), lr=0.3)
torch.manual_seed(42)

epochs = 1000
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print("X train: ", X_train)
for epoch in range(epochs):
    ### training
    model_3.train()
    y_logits = model_3(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits.squeeze(), y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # let's test!
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | "
                  f"Train loss: {loss:.5f} | "
                  f"Test loss: {test_loss:.5f} |"
                  f"Train acc: {acc:.2f}% | "
                  f"Test acc: {test_acc:.2f}%")
     
plt.title("Training Data Decision Boundary")
plot_decision_boundary(model_3, X_test, y_test)
plt.show()
