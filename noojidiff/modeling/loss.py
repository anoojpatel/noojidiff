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
    dL_dZ[target] = -1 * dL_dL #* L_curr
    return dL_dZ


#def softmax(input, i):
#   return np.exp(input[i]) / np.sum( np.exp(input)  )

def apply_softmax(input):
    return np.exp(input) / np.sum( np.exp(input)  )

def softmax_backward(dL_dNLL, P_curr, Z_curr):
    dL_dNLL
    dL_dZ = np.zeros_like(dL_dNLL)
    n = len(dL_dNLL)
    Df = np.zeros((n,n))
    for i in range(len(dL_dZ)):
        for j in range(len(dL_dZ)):
            if i != j:
                Df[j,i] =  -P_curr[i]*P_curr[j]
            else:
                Df[j,i] = -P_curr[i] + P_curr[i]**2

    for j in range(n):
        dL_dZ[j] = np.dot(Df[j],dL_dNLL)

    return dL_dZ

