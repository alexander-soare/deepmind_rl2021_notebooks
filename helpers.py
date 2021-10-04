import numpy as np


def non_deterministic_argmax(arr: np.ndarray) -> int:
    """
    Expects `arr` to be 1D
    """
    max_val = np.max(arr)
    candidates = (arr == max_val)
    return np.random.choice(
        np.arange(len(arr)), p=candidates/candidates.sum())


def softmax(arr: np.ndarray, dim=-1) -> np.ndarray:
    # First step for numerical stability, but doesn't effect the outcome
    # because softmax(x+c) = softmax(x)
    arr = arr - arr.max()  
    exp = np.exp(arr)
    res = exp / exp.sum(dim)
    return res


def simple_moving_average(arr: np.ndarray, period: int) :
    """
    Expects `arr` to be 1D
    """
    ret = np.cumsum(arr, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period
