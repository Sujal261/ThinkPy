from src.Tensor import Tensor
import numpy as np
class MSELoss:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        diff = y_pred.data - y_true.data
        mse_loss = np.mean(diff**2)
        requires_grad = y_pred.requires_grad or y_true.requires_grad
        
        def grad_fn(self):
           grad = 2 * (y_pred.data - y_true.data) / len(y_true.data)
           y_pred.backward(grad)
           
        loss_tensor = Tensor(mse_loss, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
        return loss_tensor 

class BinaryCrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        
       
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred.data, epsilon, 1 - epsilon)
        
        # Calculate BCE loss
        loss = -np.mean(
            y_true.data * np.log(y_pred_clipped) +
            (1 - y_true.data) * np.log(1 - y_pred_clipped)
        )
        
        requires_grad = y_pred.requires_grad or y_true.requires_grad
        
        def grad_fn(grad_output=1):
            if y_pred.requires_grad:
               
                grad = grad_output * (
                    (y_pred_clipped - y_true.data) /
                    (y_pred_clipped * (1 - y_pred_clipped))
                ) / len(y_true.data)
                
                y_pred.backward(grad)
        
        loss_tensor = Tensor(loss, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
        return loss_tensor
    
    
class CategoricalCrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        
        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)

        
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred.data, epsilon, 1 - epsilon)
        correct_log_probs = -np.log(y_pred_clipped[np.arange(len(y_true.data)), y_true.data.astype(int)])
        loss = np.mean(correct_log_probs)
        requires_grad = y_pred.requires_grad
        def grad_fn(grad_output=1):
            if y_pred.requires_grad:
                grad = np.zeros_like(y_pred_clipped)
                grad[np.arange(len(y_true.data)), y_true.data.astype(int)] = -1 / y_pred_clipped[np.arange(len(y_true.data)), y_true.data.astype(int)]
                grad *= grad_output / len(y_true.data)
                y_pred.backward(grad)

        loss_tensor = Tensor(loss, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
        return loss_tensor


    
