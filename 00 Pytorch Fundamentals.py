import torch

scalar = torch.tensor(7)
print(scalar.ndim) # number of dimensions
print(scalar.item()) # get the value of the tensor
vector = torch.tensor([5, 3])
print(vector.ndim) # number of dimensions

MATRIX = torch.tensor([[7, 8], [9, 10]])
print(MATRIX.shape) 

# TENSOR
TENSOR = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
print(TENSOR.ndim) # number of dimensions
print(TENSOR.shape) # shape of the tensor

TENSOR2 = torch.tensor([[[[1, 2, 3, 4], [4, 5, 6, 6], [7, 8, 9, 20]], [[1, 2, 3, 4], 
                        [4, 5, 6, 6], [7, 8, 9, 10]], [[1, 2, 3, 4], 
                        [4, 5, 6, 6], [7, 8, 9, 10]]], [[[1, 2, 3, 4], [4, 5, 6, 6], [7, 8, 9, 20]], [[1, 2, 3, 4], 
                        [4, 5, 6, 6], [7, 8, 9, 10]], [[1, 2, 3, 4], 
                        [4, 5, 6, 6], [7, 8, 9, 10]]]])

print(TENSOR2.ndim)  # number of dimensions
print(TENSOR2.shape)  # shape of the tensor

# RANDOM TENSORS
random_tensor = torch.rand(3, 4)
print(random_tensor)

# random tensor with similar shape to an image tensor
random_image_tensor = torch.rand(224, 224, 3) # height, width, color channels

print(random_image_tensor.shape) # shape of the tensor
print(random_image_tensor.ndim) # number of dimensions

# zeros and ones tensor
zeros_tensor = torch.zeros(3, 4)
print(zeros_tensor)

# ones tensor
ones_tensor = torch.ones(3, 4)
print(ones_tensor)
print(ones_tensor.dtype) # data type of the tensor

# arrange
arrange_tensor = torch.arange(1, 13, 2) # start, end, step
print(arrange_tensor.ndim)

# tensor like
ten_zeros = torch.zeros_like(arrange_tensor)
print(ten_zeros)

# float 32 tensor
float_tensor = torch.tensor([3.0, 6.0, 9.0], 
                            dtype=None,  # data type of the tensor
                            device=None,  # device of the tensor
                            requires_grad=False)  # whether to track gradients
print(float_tensor)

float_16_tensor = float_tensor.type(torch.float16) # convert to float 16
print(float_16_tensor)
multiplied = float_16_tensor * float_tensor
print(multiplied)

# matrix multiplication
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
print(torch.matmul(tensor1, tensor2))

x = torch.arange(1, 100, 10)
print(x)

print(torch.argmin(x), x.min())
print(torch.max(x), x.argmax())
print(torch.mean(x.type(torch.float32)), x.type(torch.float32).mean())

x = torch.arange(1., 10.)
x_reshaped = x.reshape(1, 9)
print(x_reshaped)

z = x.view(1,9)
print(z)
print(z.shape)

z[:, 0] = 5
print(x)

x[0] = 1

# stack tensors
x_stacked = torch.stack([x, x, x, x], dim=1)
print(x_stacked)

# TORCH SQUEEZE
x_squeezed = x_reshaped.squeeze()

print('x reshaped: ', x_reshaped)
print('x reshaped shape: ', x_reshaped.shape)
print('x squeezed: ', x_squeezed)
print('x squeezed shape: ', x_squeezed.shape)

#  unsqueeze - adds a single dimension to a target tensor at a specific 
#  dimension (dim)

x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print('x unsqueezed: ', x_unsqueezed)
print('x unsqueezed shape: ', x_unsqueezed.shape)

# permute - changes the order of dimensions in a tensor (in a specified order)
x_original = torch.rand(224, 224, 3) 

x_permuted = x_original.permute(2, 0, 1)
print('x original: ', x_original)
print('x original shape: ', x_original.shape)
print('x permuted: ', x_permuted)
print('x permuted shape: ', x_permuted.shape)

x_permuted[2, 223, 223] = 1
print(x_original[223, 223, 2]) # 1
print(x_permuted[0, :, 223]) # 1


import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(tensor)
print(array)

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random_tensor_A = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_B = torch.rand(3, 4)
print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

print(torch.cuda.is_available()) # check if GPU is available