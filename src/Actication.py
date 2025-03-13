from src.Tensor import Tensor
import numpy as np

class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, x):
        self.input = x
       
        x_data = np.clip(x.data, -500, 500)  
        self.output = 1 / (1 + np.exp(-x_data))
        requires_grad = x.requires_grad
        
        def backward(grad):
            if self.input.requires_grad:
                sigmoid_grad = grad * self.output * (1 - self.output)
                self.input.backward(sigmoid_grad)
        
        return Tensor(self.output, requires_grad=requires_grad, grad_fn=backward if requires_grad else None)


class ReLU:
    def __init__(self):
        self.input = None
        self.output= None

    def __call__(self, x):
        self.input = x
        self.output = np.maximum(0, x.data)
        output = self.output
        requires_grad = x.requires_grad

        def grad_fn(grad):
            if self.input.requires_grad:
              relu_grad =  grad*(self.input.data > 0)
              self.input.backward(relu_grad)
           
        return Tensor(output, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
    
class Softmax:
    def __init__(self):
        self.input = None
        self.output = None
    
    def __call__(self, x):
        self.input = x
        
       
        x_data = x.data - np.max(x.data, axis=1, keepdims=True)
        exp_x = np.exp(x_data)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        requires_grad = x.requires_grad
        
        def grad_fn(grad):
            if self.input.requires_grad:
                
                batch_size = self.output.shape[0]
                softmax_grad = np.zeros_like(self.output)
                
                for b in range(batch_size):
                    s = self.output[b]
                    g = grad[b]
                    
                    
                    for i in range(len(s)):
                        for j in range(len(s)):
                            if i == j:
                                softmax_grad[b, i] += g[j] * s[i] * (1 - s[i])
                            else:
                                softmax_grad[b, i] += g[j] * (-s[i] * s[j])
                
                self.input.backward(softmax_grad)
        
        return Tensor(self.output, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)


        

