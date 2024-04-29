import numpy as np
import warnings

def create_MLP(input_size: int, depth: int, width: int, class_n: int):
    """Create input layer, hidden layers, and final layer, using normal(,√1/n) where `n` is the input size of the layer

        Args:
            input_size: int input size of the MLP network
            depth: int depth of the network
            width: int number of neurons per layer
            class_n: int the number of output classes            
    """
    if depth == 1:
        raise ValueError("Cant have a MLP with depth only 1! You probably mean depth=2, where there is only 1 learnable W matrix")
    elif depth == 2:
        warnings.warn("By specifying depth 2, you will not have a hidden layer! `width` is ignored!")
        return [np.sqrt(1/input_size) * np.random.normal(size=(class_n, input_size))]
    return [np.sqrt(1/input_size) * np.random.normal(size=(width, input_size))] + \
        [np.sqrt(1/width) * np.random.normal(size=(width,width)) for _ in range(depth-2)] + \
        [np.sqrt(1/width) * np.random.normal(size=(class_n, width))]

def layer(W: np.ndarray,x: np.array):
    Z = W.dot(x)
    A = relu(Z)

    dC_dself = x 
    dC_drhs = W 
    #dL_dself = lambda dL_dC: dL_dC @ dC_dself.T
    #dL_drhs = lambda dL_dC: dC_drhs.T @ dL_dC
    return A, Z, #dL_dself, dL_drhs

def relu(Z):
    return np.maximum(0,Z)

def relu_backward(Z):
    return (Z > 0).astype(Z.dtype)

def layer_backward(dA_curr, W_curr, Z_curr, A_prev):
    #print(A_prev.shape)
    #m = A_prev.shape[1]
    
    print(f"dA_curr: {dA_curr.shape}")
    dZ_curr = dA_curr * relu_backward(Z_curr)
    print("dW_curr:", dZ_curr.shape, " @ ", A_prev.T.shape)
    dL_dW_curr = np.dot(dZ_curr, A_prev.T) 
    dL_dA_prev = np.dot(W_curr.T, dZ_curr)
    print("dA_prev:", W_curr.T.shape, " @ ", dZ_curr.shape)
    print(f"dA_prev res: {dL_dA_prev.shape}, dw_curr res:{dL_dW_curr.shape}")

    return dL_dW_curr, dL_dA_prev
