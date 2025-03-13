import numpy as np
from src.Tensor import Tensor

class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None  
    
    def step(self):
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                grad = param.grad if isinstance(param.grad, np.ndarray) else np.array(param.grad)
                param.data -= self.lr * grad

