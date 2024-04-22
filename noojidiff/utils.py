import numpy as np
import torch as t

def to_torch_with_grads(arrs: list[np.ndarray]) -> list[t.Tensor]:
    outputs = []
    for arr in arrs:
        outputs.append(t.from_numpy(arr).requires_grad_(requires_grad=True))
    return outputs