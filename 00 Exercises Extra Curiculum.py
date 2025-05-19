import torch
from torch import nn


# 2. Create a random tensor with shape (7, 7)
tensor_0 = torch.rand(7, 7)

# 3. Perform a matrix multiplication on the tensor 
# from 2 with another random tensor with shape (1, 7)
tensor_1 = torch.rand(1, 7)
print(torch.matmul(tensor_0, tensor_1.T))

# 4. Set the random seed to 0 and repeat step 2 and 3
torch.manual_seed(0)
tensor_0 = torch.rand(7, 7)
tensor_1 = torch.rand(1, 7)

print(torch.matmul(tensor_0, tensor_1.T))

# 6. Create two random tensors of shape (2, 3) and 
# send them both to the GPU (you'll need access to 
# a GPU for this). Set torch.manual_seed(1234) when 
# creating the tensors (this doesn't have to be the 
# GPU random seed).
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
#  set the model to the device
torch.manual_seed(1234)
tensor_2 = torch.rand(2, 3).to(device)
tensor_3 = torch.rand(2, 3).to(device)
print(tensor_2)
print(tensor_3)

# 7. Perform a matrix multiplication on the tensors 
# you created in 6 (again, you may have to adjust 
# the shapes of one of the tensors).

multiplied = torch.matmul(tensor_2, tensor_3.T)

# 8. Find the maximum and minimum values of the output 
# of 7.
max = multiplied.max()
min = multiplied.min()
print(f"Max: {max}, Min: {min}")

# 9. Find the maximum and minimum index values of the
#  output of 7.
max_index = multiplied.argmax()
min_index = multiplied.argmin()
print(f"Max index: {max_index}, Min index: {min_index}")

# 10. Make a random tensor with shape (1, 1, 1, 10) 
# and then create a new tensor with all the 1 dimensions
#  removed to be left with a tensor of shape (10). 
# Set the seed to 7 when you create it and print out 
# the first tensor and it's shape as well as the second
#  tensor and it's shape.

torch.manual_seed(7)
tensor_10 = torch.rand(1, 1, 1, 10)
tensor_11 = tensor_10.squeeze()
print("First tensor: ", tensor_10, "Shape: ", tensor_10.shape)
print("Second tensor: ", tensor_11, "Shape: ", tensor_11.shape)