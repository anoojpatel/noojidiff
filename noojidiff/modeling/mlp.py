import numpy as np

def layer(W: np.ndarray,x: np.array):
    return W @ x

def create_MLP(input_shape: np.array, depth: int, width: int, class_n: int):
    """Create input layer, hidden layers, and final layer"""
    return [np.zeroes_like((input_shape, width))] + [np.zeros_like((width,width)) for _ in range(depth)] + [np.zeroes_like((width,class_n))]
                                                                                                          
def relu(Z):
    return np.max(0,Z)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

