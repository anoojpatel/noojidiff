"""This implements a simple Tape based reverse-mode automatic differentiation. We are only going to implement simple operations
via ADD, MUL, EXPAND and SUM"""
import numpy as np
from typing import Optional, NamedTuple, Callable


Tape = list["TapeEntry"]


# global name reference
_name: int = 0
def fresh_name() -> str:
    """ create a new unique name for a variable: v0, v1, v2 """
    global _name
    r = f'v{_name}'
    _name += 1
    return r

class Variable:
    def __init__(self, value : np.ndarray, name: str=None):
        self.value = value
        self.name = name or fresh_name()

    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes. 
    @staticmethod
    def constant(value: np.ndarray, name: str=None):
        r = Variable(value, name)
        print(f'{r.name} = {value}')a
        return r

    def __repr__(self):
        return repr(self.value)


    # This performs a pointwise multiplication of a Variable, tracking gradients
    def __matmul__(self, rhs: 'Variable') -> 'Variable':
        return operator_matmul(self, rhs)

    def __mul__(self, rhs: 'Variable') -> 'Variable':
        # defined later in the notebook
        return operator_mul(self, rhs)

    def __add__(self, rhs: 'Variable') -> 'Variable':
        return operator_add(self, rhs)
            
    def sum(self, name: Optional[str]=None) -> 'Variable':
        return operator_sum(self, name)
    
    def expand(self, sizes: list[int]) -> 'Variable':
        return operator_expand(self, sizes)


class TapeEntry(NamedTuple):
    # names of the inputs to the original computation
    inputs : list[str]
    # names of the outputs of the original computation
    outputs: list[str]
    # apply chain rule
    propagate: 'Callable[list[Variable], list[Variable]]'


def generate_tape() -> list[TapeEntry]:
    return []

def reset_tape(tape: Tape)-> None:
    tape.clear()
    global _name
    _name = 0 # reset global varIDs

def operator_mul(self : Variable, rhs: Variable, gradient_tape: Tape) -> Variable:
    if isinstance(rhs, float) and rhs == 1.0:
        # peephole optimization
        return self

    # define forward
    r = Variable(self.value * rhs.value)
    print(f'{r.name} = {self.name} * {rhs.name}')

    # record what the inputs and outputs of the op were
    inputs = [self.name, rhs.name]
    outputs = [r.name]

    # define backprop
    def propagate(dL_doutputs: list[Variable]):
        dL_dr, = dL_doutputs
    
        dr_dself = rhs # partial derivative of r = self*rhs
        dr_drhs = self # partial derivative of r = self*rhs

        # chain rule propagation from outputs to inputs of multiply
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        dL_dinputs = [dL_dself, dL_drhs] 
        return dL_dinputs
    # finally, we record the compute we did on the tape
    gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
    return r

def operator_matmul(self : Variable, rhs: Variable, gradient_tape: Tape) -> Variable:
    """
    We can think of the pullback (vJp rule) as f(A,B)= AB => C
    Where we want to take the Jacobian of f()
     A 
    """
    if isinstance(rhs, float) and rhs == 1.0:
        # peephole optimization
        return self

    # define forward
    C = Variable(self.value @ rhs.value)
    print(f'{r.name} = {self.name} @ {rhs.name}')

    # record what the inputs and outputs of the op were
    inputs = [self.name, rhs.name]
    outputs = [r.name]

    # define backprop
    def propagate(dL_doutputs: list[Variable]):
        dL_dC, = dL_doutputs
    
        dr_dself = rhs # partial derivative of r = self*rhs
        dr_drhs = self # partial derivative of r = self*rhs

        # chain rule propagation from outputs to inputs of Matrix-matrix * matrix-vector multiplication
        dL_dself = Variable(dL_dC.value @ dr_dself.value.T) 
        dL_drhs = Variable(dr_drhs.value.T @ dL_dC.value)
        dL_dinputs = [dL_dself, dL_drhs] 
        return dL_dinputs
    # finally, we record the compute we did on the tape
    gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
    return r


def operator_add(self : Variable, rhs: Variable, gradient_tape: Tape) -> Variable:
    # Add follows a similar pattern to Mul, but it doesn't end up
    # capturing any variables.
    r = Variable(self.value + rhs.value)
    print(f'{r.name} = {self.name} + {rhs.name}')
    def propagate(dL_doutputs: list[Variable]):
        dL_dr, = dL_doutputs
        dr_dself = 1.0
        dr_drhs = 1.0
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        return [dL_dself, dL_drhs]
    gradient_tape.append(TapeEntry(inputs=[self.name, rhs.name], outputs=[r.name], propagate=propagate))
    return r

# sum is used to turn our matrices into a single scalar to get a loss.
# expand is the backward of sum, so it is added to make sure our Variable
# is closed under differentiation. Both have rules similar to mul above.

def operator_sum(self: Variable, name: Optional[str], gradient_tape: Tape) -> 'Variable':
    r = Variable(np.sum(self.value), name=name)
    print(f'{r.name} = {self.name}.sum()')
    def propagate(dL_doutputs: list[Variable]):
        dL_dr, = dL_doutputs
        size = self.value.size()
        return [dL_dr.expand(*size)]
    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate))
    return r


def operator_expand(self: Variable, sizes: list[int], gradient_tape: Tape) -> 'Variable':
    assert(self.value.dim() == 0) # only works for scalars
    r = Variable(self.value.expand(sizes))
    print(f'{r.name} = {self.name}.expand({sizes})')
    def propagate(dL_doutputs: list[Variable]):
        dL_dr, = dL_doutputs
        return [dL_dr.sum()]
    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate))
    return r



def grad(L, desired_results: list[Variable], gradient_tape:Tape) -> list[Variable]:
    # this map holds dL/dX for all values X
    dL_d : dict[str, Variable] = {}
    # It starts by initializing the 'seed' dL/dL, which is 1
    dL_d[L.name] = Variable(np.ones(()))
    print(f'd{L.name} ------------------------')

    # look up dL_dentries. If a variable is never used to compute the loss,
    # we consider its gradient None, see the note below about zeros for more information.
    def gather_grad(entries: list[str]):
        return [dL_d[entry] if entry in dL_d else None for entry in entries]

    # propagate the gradient information backward
    for entry in reversed(gradient_tape):
        dL_doutputs = gather_grad(entry.outputs)
        if all(dL_doutput is None for dL_doutput in dL_doutputs):
            # optimize for the case where some gradient pathways are zero. See
            # The note below for more details.
            continue

        # perform chain rule propagation specific to each compute
        dL_dinputs = entry.propagate(dL_doutputs)

        # Accululate the gradient produced for each input.
        # Each use of a variable produces some gradient dL_dinput for that 
        # use. The multivariate chain rule tells us it is safe to sum 
        # all the contributions together.
        for input, dL_dinput in zip(entry.inputs, dL_dinputs):
            if input not in dL_d:
                dL_d[input] = dL_dinput
            else:
                dL_d[input] += dL_dinput

    # print some information to understand the values of each intermediate 
    for name, value in dL_d.items():
        print(f'd{L.name}_d{name} = {value.name}')
    print(f'------------------------')

    return gather_grad(desired.name for desired in desired_results)