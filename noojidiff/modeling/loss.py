import numpy as np

def nll_loss(input, target):
    return -input[target]

def nll_loss_backward(dL_dL, Z_curr, target):
    """ Negative Log-likelihood Backward Pass

    We can formulate the backwards pass as the chain rule'd derivative of (-1) * input.getindex(target)
    Specifically, the backwards pass of getindex is setindex(∂L)
        Args:
            dL_dL: The Cotangent Output
            Z_curr: primal input
            target: primal input
    """
    dL_dZ = np.ones_like(Z_curr)
    dL_dZ[target] = -1 * dL_dL
    return dL_dZ
