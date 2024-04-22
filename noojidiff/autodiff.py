"""This implements a simple Tape based reverse-mode automatic differentiation. We are only going to implement simple operations
via ADD, MUL, EXPAND and SUM"""
import numpy as np
import itertools
from typing import Optional, NamedTuple, Callable, Any
from contextlib import contextmanager

Tape = list["TapeEntry"]

tapes: list[Tape] = []

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
        print(f'{r.name} = {value}')
        return r

    def __repr__(self):
        return repr(self.value)

    def get_tape(self):
        if tapes:
            return tapes[-1]
        else:
            raise KeyError("Gradient Tape not defined in scope. Make sure you use a context manager.")
    # This performs a pointwise multiplication of a Variable, tracking gradients
    def __matmul__(self, rhs: 'Variable') -> 'Variable':

        return operator_matmul(self, rhs, self.get_tape())

    def __mul__(self, rhs: 'Variable') -> 'Variable':
        # defined later in the notebook
        return operator_mul(self, rhs, self.get_tape())

    def __add__(self, rhs: 'Variable') -> 'Variable':
        return operator_add(self, rhs, self.get_tape())
    
    def __neg__(self) -> 'Variable':
        return operator_mul(self, Variable.constant(-1, name="const(-1)"), self.get_tape())
    
    @property
    def T(self) -> 'Variable':
        return operator_transpose(self,False, self.get_tape())
    
    @property
    def conj(self) -> 'Variable':
        return operator_transpose(self,True, self.get_tape())

    def __getitem__(self, index: Any) -> 'Variable':
        print(f"get_item called: {index} on self:{self.name}:{self.value}")
        return operator_getindex(self, index, self.get_tape())

    def sum(self, name: Optional[str]=None) -> 'Variable':
        return operator_sum(self, name, self.get_tape())
    
    def expand(self, sizes: list[int]) -> 'Variable':
        return operator_expand(self, sizes, self.get_tape())
    
    def reshape(self, new_shape: tuple) -> 'Variable':
        return operator_reshape(self, new_shape, self.get_tape())

PrimalVariableInputs = list[str]
PrimalVariableOutputs = list[str]

TangentOutputs = list[Variable]
TangentInputs = list[Variable]

class TapeEntry(NamedTuple):
    # names of the inputs to the original computation (Primal Inputs)
    inputs : PrimalVariableInputs
    # names of the outputs of the original computation (Primal Outputs)
    outputs: PrimalVariableOutputs
    # apply chain rule
    propagate: Callable[[TangentOutputs], TangentInputs]


@contextmanager
def tape_context(add_new_tape: bool=False):
    if add_new_tape:
        tapes.append(generate_tape())
    try:
        yield tapes[-1]
    finally:
        reset_tape(tapes[-1]) 

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
        print(dL_dr) 
        dr_dself = rhs # partial derivative of r = self*rhs
        dr_drhs = self # partial derivative of r = self*rhs

        # chain rule propagation from outputs to inputs of multiply
        print("dL_dself = dL_dr * dr_dself -> ", dL_dr, " * ", dr_dself)
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        dL_dinputs = [dL_dself, dL_drhs] 
        return dL_dinputs
    # finally, we record the compute we did on the tape
    gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
    return r

def operator_matmul(self : Variable, rhs: Variable, gradient_tape: Tape) -> Variable:
    """
    We can think of the pullback (vJp rule) as f(Self,Rhs)= Self @ Rhs => C
    Where we want to take the Jacobian of f()
     A 
    """
    if isinstance(rhs, float) and rhs == 1.0:
        # peephole optimization
        return self

    # define forward
    C = Variable(self.value @ rhs.value)
    print(f'{C.name} = {self.name} @ {rhs.name}')

    # record what the inputs and outputs of the op were
    inputs = [self.name, rhs.name]
    outputs = [C.name]

    # define backprop
    def propagate(dL_doutputs: list[Variable]):
        dL_dC, = dL_doutputs
        print(dL_dC, dL_dC.value.ndim == 0)
        dC_dself = rhs # partial derivative of r = self*rhs
        dC_drhs = self # partial derivative of r = self*rhs

        # Check if primal output is a scalar / zero dimmed scalar overload
        if  C.value.ndim == 0 or np.isscalar(C.value):
            dL_dC.value = dL_dC.value * np.ones_like(self.value @ rhs.value)
        # chain rule propagation from outputs to inputs of Matrix-matrix * matrix-vector multiplication
        print("dL_dself = dL_dC * dr_dself -> ", dL_dC, " @ ", dC_dself)

        # Quite brutal logic to handle casting the right operation for combiantions of primal inputs,
        # primal outputs, cotangent inputs and outputs dimenisons w.r.t. a matrix mul in the primal

        #Check if cotangent output being pulledback is scalar
        if dL_dC.value.ndim == 0 or np.isscalar(dL_dC.value):
            if self.value.ndim <= 1:
                print(self.value.ndim, "matmul self dim")
                dL_dself = Variable(dL_dC.value * dC_dself.value)
            else:
                dL_dself = Variable(dL_dC.value * (dC_dself.value.T @ np.ones_like(dL_dC.value)))

            if rhs.value.ndim <= 1:
                print(rhs.value.ndim, "matmul rhs dim")
                dL_drhs = Variable(dL_dC.value * dC_drhs.value)
            else:
                dL_drhs = Variable(dL_dC.value * ( dC_drhs.value.T @ np.ones_like(dL_dC.value))) 
        else:
            dL_dself = Variable(dL_dC.value @ dC_dself.value.T) 
            dL_drhs = Variable(dC_drhs.value.T @ dL_dC.value)
        dL_dinputs = [dL_dself, dL_drhs] 
        return dL_dinputs
    # finally, we record the compute we did on the tape
    gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
    return C


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
    print("expand on: ", self.value, self.name)
    assert(self.value.ndim == 0) # only works for scalars
    r = Variable(np.expand_dims(self.value,axis=sizes))
    print(f'{r.name} = {self.name}.expand({sizes})')
    def propagate(dL_doutputs: list[Variable]):
        dL_dr, = dL_doutputs
        return [dL_dr.sum()]
    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate))
    return r

def operator_getindex(self: Variable, indices: list[int], gradient_tape: Tape) -> Variable:
    """ Based off ChainRules.jl reverse-rule https://github.com/JuliaDiff/ChainRules.jl/blob/v0.7.49/src/rulesets/Base/indexing.jl"""
    print(f" operator_getindex: indices: {indices}, value: ", self.value)
    if isinstance(indices, int):
        # Handle scalar indexing case
        y = Variable(self.value[indices])
    else:
        y = Variable(self.value[indices.value.tolist()])

    def propagate(dl_doutputs: list[Variable]):
        dL_dr, = dl_doutputs
        dr = np.zeros_like(self.value)
        if np.isscalar(dL_dr) or dL_dr.value.ndim == 0:
            for  ii in itertools.product(indices.value):
                print("ii value: ",type(ii[0].value))
                dr[ii[0]] += dL_dr.value
        else:
            print("indices: ", indices)
            for y_ii, ii in zip(dL_dr.value, itertools.product(indices.value)):
                print("ii value: ",ii, " y_ii value: ",y_ii, "dr: ", dr)
                dr[ii] += y_ii

        print("getindex value: ", dr, dL_dr.value)
        return [Variable(dr)]
    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[y.name], propagate=propagate))
    return y


def difreshape(sizea, numela, numelc, f):
    """ Method we can use to handle reshapes for symbolic forwards and pullbacks for most reshape calls
    Note, this will need to be combined with another sparsity based helper method to handle strides effeciently
    for CPU-RAM based np.arrays

    Args:
        sizea: tuple size of the original array
        numela: int number of original elmenents
        numelc: int number of final elements
        f: Callable function that maps indicies of the original array to that of the resulting array
    """
    aidx = np.arange(1, numela + 1).reshape(sizea[::-1]).T
    aidx = f(aidx)
    aidx_flat = aidx.flatten('F')
    aeq = np.nonzero(aidx_flat)[0]
    row = aeq
    col = aidx_flat[aeq]
    da = np.zeros((numelc, numela))
    da[row, col - 1] = 1
    return da, row, col


def operator_reshape(self: Variable, new_shape: tuple, gradient_tape: Tape) ->'Variable':
    reshaped = Variable(np.reshape(self.value, new_shape))

    def propagate(dl_outputs: TangentOutputs) ->TangentInputs:
        dL_dr, = dl_outputs
        dL_dself = difreshape(self.value.shape, np.prod(new_shape), np.prod(self.value.shape), lambda x: x)
        return [Variable(np.dot(dL_dr, dL_dself))]
    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[reshaped.name], propagate=propagate))
    return reshaped

def boxproduct(a, b):
    return np.kron(np.ones((1, b.shape[1])), np.kron(a, np.ones((b.shape[0], 1)))) * np.kron(np.kron(np.ones((a.shape[0], 1)), b), np.ones((1, a.shape[1])))

    
def operator_transpose(self: Variable, complex: bool, gradient_tape: Tape)-> 'Variable':
    if complex:
        y = Variable(self.value.conj)
    else:
        y = Variable(self.value.T)
    def propagate(dl_outputs: TangentOutputs) ->TangentInputs:
        dL_dr, = dl_outputs
        dL_dself = Variable.boxproduct(np.eye(self.value.shape[1]), np.eye(self.value.shape[0])).toarray()
        return [Variable(np.dot(dL_dr.value, dL_dself.value))]

    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[y.name], propagate=propagate))

def grad(L, desired_results: list[Variable], gradient_tape:Tape) -> list[Variable]:
    # this map holds dL/dX for all values X
    dL_d : dict[str, Variable] = {}
    # It starts by initializing the 'seed' dL/dL, which is 1
    dL_d[L.name] = Variable(np.ones(()))
    print(f'd{L.name} {type(L.value)} {L.value.shape if isinstance(L.value,np.ndarray) else ""}------------------------')

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