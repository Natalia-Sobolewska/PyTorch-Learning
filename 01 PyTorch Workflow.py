import torch
from torch import nn
import matplotlib.pyplot as plt


class LinearRegressionModel(nn.Module):
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


class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, 
                                       out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x) 
    

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
    plt.show()

torch.manual_seed(42)  # for reproducibility
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))  
print("Model 0: ", model_0.state_dict())  # state_dict is a Python dictionary object that maps each layer to its parameter tensor

with torch.inference_mode():
    y_preds = model_0(X_test)
    # print(y_preds[:10])
    # plot_predictions(predictions=y_preds)  # plot predictions

loss_fn = nn.L1Loss()  # L1 loss function
print("Loss fn: ",loss_fn)  # loss function test
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                              lr=0.01)  # stochastic gradient descent optimizer

# building a training (and a testing loop) loop in pyTorch
# 0. loop through the data
# 1. forward pass (data moving through the model's forward method) - 
# forward propagation
# 2. calculate the loss (how far off the model's predictions 
# are from the target labels)
# 3. optimizer zero grad (clear the gradients)
# 4. backward pass (backpropagation) - move backwards through the network
# to calculate the gradients of each of the parameters with respect to the loss
# 5. optimizer step - use the optimizer to adjust the parameters 
# to improve the loss (gradient descent)

# epoch is one loop through the data - hyperparameter (because we set it - ourselves)
epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []

# loop through the data
for epoch in range(epochs):
    #  set model to training mode
    model_0.train() # sets all the parameters that require gradients 
                    # to require gradients
    # forward pass
    y_pred = model_0(X_train)  # pass the training data through the model
    # calculate the loss
    loss = loss_fn(y_pred, y_train)  # compare the predictions to the training labels

    # optimizer zero grad
    optimizer.zero_grad() 
    # backprogagation
    loss.backward()  # calculate the gradients of the loss with respect to the parameters
    # optimizer step
    optimizer.step()  # update the parameters using the gradients

    model_0.eval() # turns off the gradients tracking
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)
        #  plot_predictions(predictions=test_pred)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())  # print the model's parameters

# plot the loss curves
plt.plot(epoch_count, loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Loss curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print(loss_values)
with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)
        plot_predictions(predictions=test_pred)

# save and load the model
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)  # create the directory if it doesn't exist

MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(model_0.state_dict(), MODEL_SAVE_PATH)  # save the model

# load the model
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model.eval()  # set the model to evaluation mode
with torch.inference_mode():
    loaded_model_preds = loaded_model(X_test)
    print(loaded_model_preds == test_pred)

torch.manual_seed(42)  # for reproducibility
model_1 = LinearRegressionModel()
print(list(model_1.parameters()))  
print("Model 1", model_1.state_dict())  # state_dict is a Python dictionary object that maps each layer to its parameter tensor

with torch.inference_mode():
    y_preds = model_1(X_test)
    # print(y_preds[:10])
    plot_predictions(predictions=y_preds)  # plot predictions

"""
# to set the model to a deivce (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
#  set the model to the device
model_0.to(device)
next(model_0.parameters()).device  # check the device of the model's parameters

# remember to set the data to the same device as the model
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
"""