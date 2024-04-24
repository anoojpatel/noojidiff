import numpy as np

def nll_loss(input, target):
    return -input[target]

def nll_loss_backward(dL_dL, L_curr, Z_curr, target):
    """ Negative Log-likelihood Backward Pass

    We can formulate the backwards pass as the chain rule'd derivative of (-1) * input.getindex(target)
    Specifically, the backwards pass of getindex is setindex(∂L)
        Args:
            dL_dL: The Cotangent Output
            Z_curr: primal input
            target: primal input
        Returns:
            Cotangents inputs to the logits
    """
    dL_dZ = np.zeros_like(Z_curr)
    print(dL_dZ.shape)
    dL_dZ[target] = -1 * dL_dL * L_curr
    return dL_dZ
