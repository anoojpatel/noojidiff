import numpy as np
from noojidiff.modeling.loss import nll_loss
from noojidiff.modeling.mlp import create_MLP, layer, relu, relu_backward, layer_backward

mlp = create_MLP(4, 2, 5, 3)
x = np.random.rand(4)
activation = x
activations = []
#print(mlp)
for W in mlp:
    activations.append(activation:= layer(W,activation))

print([activation.shape for activation in activations])