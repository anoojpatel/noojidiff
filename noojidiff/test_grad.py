import numpy as np
from autodiff import *

a_global, b_global = np.random.rand(4), np.random.rand(4)

with tape_context(add_new_tape=True) as tape:
    def simple(a, b):
        t = a + b
        return t @ b


    a = Variable.constant(a_global, name='a')
    b = Variable.constant(b_global, name='b')
    loss = simple(a, b)
    da, db = grad(loss, [a, b], tape)
    print("da", da)
    print("db", db)
    print(tape)