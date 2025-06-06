import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import requests
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from pathlib import Path
from torchmetrics import Accuracy

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download.")
else:
    print("Downloading helper_functions.py...")
    url = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
    response = requests.get(url)
    with open("helper_functions.py", "wb") as f:
        f.write(response.content)
    print("Download complete.")

from helper_functions import plot_predictions, plot_decision_boundary, accuracy_fn

# set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42


class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes multiclass classification model.
        Args:
            input_features (int): Number of input features to the model
            output_features (int): Number of outputs (number of output classes)
            hidden_units (int): Number of hidden units between layers, default 8
            
        Returns:
        
        Examples:
        """

        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)


# create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

# turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float32)
y_blob = torch.from_numpy(y_blob).type(torch.long)

# split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, 
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# visualize visualize visualize
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()

# create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_0 = BlobModel(input_features=NUM_FEATURES,
                    output_features=NUM_CLASSES,
                    hidden_units=8).to(device)
print(model_0)

# create a loss function for multi-class classification
loss_fn = nn.CrossEntropyLoss()
# create an optimizer for multi-class classification
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# getting prediction probabilities for a multi-class PyTorch model
print(model_0(X_blob_test))

model_0.eval()

with torch.inference_mode():
    y_logits = model_0(X_blob_test.to(device))

y_pred_probs = torch.softmax(y_logits, dim=1)
print(f"Predicted logits:\n{y_logits[:10]}")
print(f"Predicted probabilities:\n{y_pred_probs[:10]}")

y_preds = torch.argmax(y_pred_probs, dim=1)

print(f"Predicted class labels:\n{y_preds[:10]}")

### create a training loop and a testing loop 
torch.manual_seed(RANDOM_SEED)

epochs = 100

X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_blob_train)
    y_pred_probs = torch.softmax(y_logits, dim=1)
    y_preds = torch.argmax(y_pred_probs, dim=1)
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_preds)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | "
                  f"Train loss: {loss:.5f} | "
                  f"Train acc: {acc:.2f}% | "
                  f"Test loss: {test_loss:.5f} | "
                  f"Test acc: {test_acc:.2f}%")

# visualize
plot_decision_boundary(model_0, X_blob, y_blob)
plt.show()

torchmetrics_acc = Accuracy(task="multiclass", num_classes=4).to(device)
print(torchmetrics_acc(test_preds, y_blob_test))