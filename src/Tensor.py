import numpy as np 
class Tensor:
    def __init__(self, data, requires_grad = False , grad_fn = None ):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None 
        self.grad_fn = grad_fn 
        
    def backward(self, grad= None):
        
        if not self.requires_grad:
            return
        
        
        if grad is None:
            grad = np.ones_like(self.data)
            
        if self.grad is None:
            self.grad = grad
        else:
            self.grad+=grad
            
        if self.grad_fn is not None:
            self.grad_fn(self.grad)
            
            
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        
        def grad_fn(grad):
            if self.requires_grad:
                grad_self = grad if grad.shape == self.data.shape else np.mean(grad, axis=0, keepdims=True)
                self.backward(grad_self)
            if other.requires_grad:
                grad_other = grad if grad.shape == other.data.shape else np.mean(grad, axis=0, keepdims=True)
                other.backward(grad_other)
        return Tensor(data, requires_grad, grad_fn if requires_grad else None)
                
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data-other.data
        requires_grad = self.requires_grad or other.requires_grad
        
        def grad_fn(grad):
            if self.requires_grad:
                grad_self = grad if grad.shape == self.data.shape else np.mean(grad, axis=0, keepdims=True)
                self.backward(grad_self)
            if other.requires_grad:
                grad_other = -grad if grad.shape == other.data.shape else np.mean(-grad, axis=0, keepdims=True)
                other.backward(grad_other)
        return Tensor(data, requires_grad , grad_fn if requires_grad else None)
        
                
        
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data*other.data
        requires_grad = self.requires_grad or other.requires_grad
        
        def grad_fn(grad):
            if self.requires_grad:
                grad_self = grad * other.data if grad.shape == self.data.shape else np.mean(grad * other.data, axis=0, keepdims=True)
                self.backward(grad_self)
            if other.requires_grad:
                grad_other = grad * self.data if grad.shape == other.data.shape else np.mean(grad * self.data, axis=0, keepdims=True)
                other.backward(grad_other)
                
        return Tensor(data, requires_grad, grad_fn if requires_grad else None)
                
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = np.dot(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        
        
        def grad_fn(grad):
            if self.requires_grad:
                grad_self = np.dot(grad, other.data.T)
                self.backward(grad_self)
            if other.requires_grad:
                grad_other = np.dot(self.data.T, grad)
                other.backward(grad_other)
                
        return Tensor(data, requires_grad, grad_fn if requires_grad else None)
    
    
    def __pow__(self, power):
        data = self.data**power
        requires_grad = self.requires_grad
        
        def grad_fn(grad):
            if self.requires_grad:
                self.backward(grad*power*self.data**(power - 1))
        return Tensor(data, requires_grad, grad_fn if requires_grad else None)
    
    def __truediv__(self, other):
        data = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data/other.data
        requires_grad = self.requires_grad or other.requires_grad
        
        
        def grad_fn(grad):
            if self.requires_grad:
                self.backward(grad/other.data)
                
            if other.requires_grad:
                other.backward(-grad*self.data/(other.data**2))
                
        return Tensor(data , self.requires_grad, grad_fn if requires_grad else None )
                
    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    
