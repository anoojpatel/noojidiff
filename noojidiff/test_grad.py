import numpy as np
from autodiff import *

a_global, b_global, c_global = np.random.rand(4), np.random.rand(4), np.random.rand(4)

with tape_context(add_new_tape=True) as tape:
    def simple(a, b, c):
        t = a + b
        t = a * c
        return t @ b


    a = Variable.constant(a_global, name='a')
    b = Variable.constant(b_global, name='b')
    c = Variable.constant(c_global, name='c')
    loss = simple(a, b, c)
    da, db = grad(loss, [a, b], tape)
    print("da", da)
    print("db", db)
    print(tape)