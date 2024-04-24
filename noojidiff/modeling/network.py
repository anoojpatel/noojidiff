import numpy as np
from noojidiff.modeling.loss import nll_loss, nll_loss_backward
from noojidiff.modeling.mlp import create_MLP, layer, layer_backward

mlp = create_MLP(4, 4, 5, 3) # can clean up later to make depth calculation be cleaner, but here 2 is hidden depth
x = np.random.rand(4,1) # single 4 dim'd input
y = np.array([2]) # Class label 2 for our example


def forward(mlp, x):
    activation = x 
    activations = {}
    activations_counter = 0
    activations[activations_counter] = activation
    activations_counter +=1
    logits = {}
    counter = 0
    for W in mlp:
        print(f"x_in: {activation}")
        activation, logit = layer(W, activation)
        activations.update({activations_counter: activation}), logits.update({counter:logit})
        print( f" A_{counter}:{activation}, W:{W.shape}")
        counter += 1
        activations_counter += 1
    return activations, logits
    
activations, logits = forward(mlp, x)

print("mlp depth:",len(mlp))
y_hat = nll_loss(activations[3], y)
activations.update({"y_hat": y_hat})
#print("logits: ",logits)
print(f"y_hat: {y_hat}")


# Compute backwards passes
dL_dL = 1.0 # We seed the backwards pass, because of the chain-rule, ∂L/∂L = 1


def full_mlp_backpropagate(mlp, activations, logits, y):
    dL_dL = 1.0
    # compute derivative of activations w.r.t. nll_loss
    dA = nll_loss_backward(dL_dL, activations["y_hat"], activations[len(mlp)], y)
    cotangents = [dA]
    print(f"==== y_hat: {dA.shape}=======")
    weight_grads = {}
    for i in reversed(range(len(mlp))):
        l, curr_logits, curr_activations = mlp[i], logits[i], activations[i]
        assert dA.shape == curr_logits.shape, f"{dA.shape} != {curr_logits.shape}"
        dA_W, dA = layer_backward(dA, l, curr_logits, curr_activations)
        cotangents.append(dA)
        if i in weight_grads:
            weight_grads[i] += dA_W
        else:
            weight_grads[i] = dA_W

    return weight_grads, cotangents
        

weight_grads,cots = full_mlp_backpropagate(mlp, activations, logits, y)
print("weights:", [weight_grads[i].shape for i in reversed(range(4))])
print("cotangents: ",[cot.shape for cot in cots])
print("activations: ", [activations[i].T.shape for i in reversed(range(4))])
