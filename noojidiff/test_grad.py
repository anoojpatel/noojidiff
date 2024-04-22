import numpy as np
import torch as t
from autodiff import *
from utils import to_torch_with_grads
from modeling.loss import nll_loss

a_global, b_global, c_global = np.random.rand(4), np.random.rand(4), np.random.rand(4)
a_t, b_t, c_t = to_torch_with_grads([a_global.copy(), b_global.copy(), c_global.copy()])
# print(a_t,b_t, c_t)
with tape_context(add_new_tape=True) as tape:
    def simple(a, b, c):
        t = a + b
        t = a * c
        return t @ b

    a = Variable.constant(a_global, name='a')
    b = Variable.constant(b_global, name='b')
    c = Variable.constant(c_global, name='c')
    loss = simple(a, b, c)
    print("noojidff primal output: ", loss)
    print("torch primal output: ", t_loss := simple(a_t, b_t, c_t))
    t_seed = t.ones(())
    t_loss.backward(gradient=t_seed)
    da, db, dc = grad(loss, [a, b, c], tape)
    print("da", da)
    print("db", db)
    print("dc", dc)
    assert np.allclose(a_t.grad.numpy(), da.value) 
    assert np.allclose(b_t.grad.numpy(), db.value) 
    assert np.allclose(c_t.grad.numpy(), dc.value) 
    print(tape)

inputs = np.random.rand(2)
target = np.zeros((1,1),dtype=np.int64)
t_inputs, = to_torch_with_grads([inputs.copy().reshape((1,2))])
t_targets = t.from_numpy(np.zeros(1, dtype=np.int64))
with tape_context(add_new_tape=True) as tape:
    inp = Variable.constant(inputs, name="inputs")
    tgt = Variable.constant(target, name="target")
    def nll_loss(input, target):
        print("type of input: ",type(input))
        return Variable.constant(np.array([-1]), name="const(-1)") * (input[target])
    loss = nll_loss(inp, tgt)
    t_loss = t.nn.functional.nll_loss(t_inputs, t_targets.requires_grad_(requires_grad=False))
    t_seed = t.ones(())
    t_loss.backward(gradient=t_seed)
    dinp, dtgt = grad(loss, [inp, tgt], tape)
    print("dinp ",dinp)
    print("dtgt ",dtgt)
    print("t_inputs: ",t_inputs.grad)
    print("t_targets: ",t_targets.grad)
    
