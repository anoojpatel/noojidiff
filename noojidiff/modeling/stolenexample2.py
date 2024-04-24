""" Generated example from chatGPT to also help with debugging"""

import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(z.dtype)

def forward(x, W, b):
    z = np.dot(W, x) + b
    a = relu(z)
    return z, a

def backward_pass(dL_da, z, W_prev, x):
    print(f"dl_da.shape: {dL_da.shape} * {relu_derivative(z).shape}, {z.shape}")
    dL_dz = dL_da * relu_derivative(z)
    dL_dW = np.dot(dL_dz, x.T)
    dL_db = np.sum(dL_dz, axis=1, keepdims=True)
    dL_da_prev = np.dot(W_prev.T, dL_dz)
    return dL_dW, dL_db, dL_da_prev

# Initializing weights and biases for a simple 3-layer MLP
np.random.seed(0)  # for reproducibility
input_size = 4
hidden_size1 = 5
hidden_size2 = 3
output_size = 2

W1 = np.random.randn(hidden_size1, input_size)
b1 = np.zeros((hidden_size1, 1))

W2 = np.random.randn(hidden_size2, hidden_size1)
b2 = np.zeros((hidden_size2, 1))

W3 = np.random.randn(output_size, hidden_size2)
b3 = np.zeros((output_size, 1))

# Example input vector
x = np.random.randn(input_size, 1)

# Forward pass through the network
z1, a1 = forward(x, W1, b1)
z2, a2 = forward(a1, W2, b2)
z3, a3 = forward(a2, W3, b3)  # Assuming no activation in the output layer for simplicity

# Assume some gradient coming from the loss function at the output
# Here we use a simple example where the gradient is just the output error assuming a simple L2 loss
y_true = np.random.randn(output_size, 1)  # true values for the output
dL_da3 = 2 * (a3 - y_true)  # derivative of L2 loss
print(f"dl_loss shape {dL_da3.shape}, z3:{z3.shape}, a2:{a2.shape}")
# Backward pass through each layer
dL_dW3, dL_db3, dL_da2 = backward_pass(dL_da3, z3, W3, a2)
dL_dW2, dL_db2, dL_da1 = backward_pass(dL_da2, z2, W2, a1)
dL_dW1, dL_db1, _ = backward_pass(dL_da1, z1, W1, x)

# Outputs to see the gradients
print("Gradient w.r.t. Weights in Layer 1:\n", dL_dW1)
print("Gradient w.r.t. Biases in Layer 1:\n", dL_db1)
print("Gradient w.r.t. Weights in Layer 2:\n", dL_dW2)
print("Gradient w.r.t. Biases in Layer 2:\n", dL_db2)
print("Gradient w.r.t. Weights in Layer 3:\n", dL_dW3)
print("Gradient w.r.t. Biases in Layer 3:\n", dL_db3)