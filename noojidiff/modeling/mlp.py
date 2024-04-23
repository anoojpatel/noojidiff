import numpy as np

def create_MLP(input_size: int, depth: int, width: int, class_n: int):
    """Create input layer, hidden layers, and final layer, using normal(,√1/n) where `n` is the input size of the layer

        Args:
            input_size: int input size of the MLP network
            depth: int depth of the network
            width: int number of neurons per layer
            class_n: int the number of output classes            
    """
    return [np.sqrt(1/input_size) * np.random.normal(size=(width, input_size))] + \
        [np.sqrt(1/width) * np.random.normal(size=(width,width)) for _ in range(depth)] + \
        [np.sqrt(1/width) * np.random.normal(size=(class_n, width))]

def layer(W: np.ndarray,x: np.array):
    Z = W @ x
    A = relu(Z)
    return A

def relu(Z):
    return np.maximum(0,Z)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def layer_backward(dA_curr, W_curr, Z_curr, A_prev):
    m = A_prev.shape[1]
    
    backward_activation_func = relu_backward
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = (dZ_curr @ A_prev.T) / m
    dA_prev = (W_curr.T @ dZ_curr)

    return dA_prev, dW_curr