import numpy as np
import torch

def np_mul(val1, val2):
    val1 = np.asarray(val1)
    val2 = np.asarray(val2)
    print(f"type of values: {val1.dtype, val2.dtype}")
    print(f"shape of val1 : {val1.shape}")
    print(f"shape of val2 : {val2.shape}")

    out = val1.dot(val2)
    print(f"dot prod: \n {out}")
    print(f"shape of dot prod: {out.shape}")

def tensor_mul(val1,val2):
    val1 = torch.tensor(val1)
    val2 = torch.tensor(val2)
    print(f"type of values: {val1.dtype, val2.dtype}")
    print(f"shape of val1 : {val1.shape}")
    print(f"shape of val2 : {val2.shape}")

    out = val1 @ val2
    print(f"dot prod: \n {out}")
    print(f"shape of dot prod: {out.shape}")