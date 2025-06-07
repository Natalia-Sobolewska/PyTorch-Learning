import torch
import matplotlib.pyplot as plt
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, 
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, 
                                                dtype=torch.float))
    def forward(self, x):
        return self.weight * x + self.bias
    

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, 
                                                requires_grad=True, 
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True, 
                                             dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias  #linear regression formula



"""
1. Create a straight line dataset using the linear regression formula (weight * X + bias).
Set weight=0.3 and bias=0.9 there should be at least 100 datapoints total.
Split the data into 80% training, 20% testing.
Plot the training and testing data so it becomes visual
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)  # for reproducibility

weight = 0.3
bias = 0.9
start = 0
end = 10
step = 0.1
X = torch.arange(start, end, step).unsqueeze(dim=1)  # shape (99, 1)
y = weight * X + bias

# splitting the data into training and testing sets
train_border = int(0.8 * len(X))
X_train, X_test = X[:train_border], X[train_border:]
y_train, y_test = y[:train_border], y[train_border:]

# plot the training and testing data
plt.figure(figsize=(10, 7))
plt.title("Training and Testing Data")
plt.scatter(X_train, y_train, c="b", label="Train data")
plt.scatter(X_test, y_test, c="g", label="Test data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

model_0 = LinearRegressionModelV2().to(device)
print(model_0.state_dict())

# create a loss function and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# put the data to the device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 200

for epoch in range(epochs):
    model_0.train()

    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_preds = model_0(X_test)
        test_loss = loss_fn(test_preds, y_test)
        if epoch % 20 == 0:
            print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# plot the predictions
plt.figure(figsize=(10, 7))
plt.title("Training and Testing Data")
plt.scatter(X_train, y_train, c="b", label="Train data")
plt.scatter(X_test, y_test, c="g", label="Test data")
plt.scatter(X_test, test_preds, c="r", label="Model predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# save the model
torch.save(model_0.state_dict(), "linear_regression_model.pth")
# load the model
model_1 = LinearRegressionModel().to(device)
model_1.load_state_dict(torch.load("linear_regression_model.pth"))
# check if the loaded model is the same as the original model
print(f"Model 0 state dict: {model_0.state_dict()}")
print(f"Model 1 state dict: {model_1.state_dict()}")
print(f"Are the models equal? {model_0.state_dict() == model_1.state_dict()}")