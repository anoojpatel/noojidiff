import numpy as np
from noojidiff.modeling.loss import nll_loss, nll_loss_backward
from noojidiff.modeling.mlp import create_MLP, layer, layer_backward
from noojidiff.utils import to_torch_with_grads
mlp = create_MLP(4, 2, 5, 3) # can clean up later to make depth calculation be cleaner, but here 2 is hidden depth
x = np.random.rand(4,1) # single 4 dim'd input
y = np.array([0]) # Class label 0 for our example


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
loss = nll_loss(activations[len(mlp)], y)
activations.update({"loss": loss})
#print("logits: ",logits)
print(f"y_hat: {activations[len(mlp)]}")



def full_mlp_backpropagate(mlp, activations, logits, y):
    dL_dL = 1.0 # We seed the backwards pass, because of the chain-rule, ∂L/∂L = 1
    # compute derivative of activations w.r.t. nll_loss
    dA = nll_loss_backward(dL_dL, activations[len(mlp)], activations[len(mlp)], y)
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
print("weights:", [weight_grads[i].shape for i in reversed(range(len(mlp)))])
print("cotangents: ",[cot.shape for cot in cots])
print("activations: ", [activations[i].T.shape for i in reversed(range((len(mlp))))])

print(weight_grads)
print("cotagents: ", cots)

# torch verfication for 1 hidden layer network
import torch 
torchW, = to_torch_with_grads([mlp[0].copy()])
t_inputs, = to_torch_with_grads([x.copy().reshape((4,1))])
t_targets = torch.from_numpy(np.zeros(1, dtype=np.int64))
z = torchW.matmul(t_inputs)
#z_test = torch.rand(3,1).requires_grad_(requires_grad=True)
a = torch.nn.functional.relu(z)
t_loss = torch.nn.functional.nll_loss(a.reshape(1,3), t_targets.requires_grad_(requires_grad=False))
print("t_loss: ", t_loss)
t_seed = torch.ones(())
t_loss.backward(gradient=t_seed)
print("torch activations: ", )
print("W gradients:", torchW.grad)
print("input gradients: ",t_inputs.grad)
#print("a gradients: ",z_test.grad)