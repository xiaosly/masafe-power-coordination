"""Continuous-space descriptors consumed by the policy and rollout buffer."""
import numpy as np


class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.low = np.full(self.shape, low, dtype=self.dtype)
        self.high = np.full(self.shape, high, dtype=self.dtype)

    def __repr__(self):
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype})"
