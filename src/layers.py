from src.Tensor import Tensor
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0/in_features)
        self.weights = Tensor(np.random.randn(in_features, out_features)*scale, requires_grad = True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad = True)
        
    def __call__(self,x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.x = x
        output=x.data@self.weights.data+self.bias.data
        requires_grad = self.weights.requires_grad or self.bias.requires_grad or x.requires_grad
        
        def grad_fn(grad):
            if self.weights.requires_grad:
                self.weights.grad = x.data.T @ grad
            if self.bias.requires_grad:
                self.bias.grad = np.sum(grad, axis = 0, keepdims=True)
            if x.requires_grad:
                x.backward(grad @ self.weights.data.T)
        return Tensor(output, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
   
        
    def parameters(self):
        return [self.weights, self.bias]
    
    
class Flatten:
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, x):
        self.input = x
        p =x.data
        self.output = p.reshape(p.shape[0], -1)
        requires_grad = x.requires_grad
        
        def grad_fn(grad):
            if x.requires_grad:
                flatten_grad = grad.reshape(x.data.shape)
                self.input.backward(flatten_grad)
        return Tensor(self.output, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)

    
class MaxPool:
    def __init__(self, stride, kernel_size):
        self.input = None 
        self.output = None
        self.stride = stride
        self.kernel_size = kernel_size
        
    def __call__(self, x ):
        self.input = x
        e, f = self.kernel_size[0], self.kernel_size[1]
        batch_size,channels, i, j = x.data.shape
        
        g,h = (i-e)//self.stride+1, (j-f)//self.stride+1
        output = np.zeros((batch_size, channels, g, h))
        max_indices = np.zeros((batch_size, channels, g, h, 2))
        for k in range(batch_size):
            for c in range(channels):
                for a in range(g):
                    
                    for b in range(h):
                        h_start = a*self.stride
                        w_start = b*self.stride
                        window = x.data[k, c, h_start:h_start+e, w_start:w_start+f]
                        max_value = np.max(window)
                        flat_index = np.argmax(window)
                        max_row, max_col = np.unravel_index(flat_index, window.shape)
                        output[k,c,a,b] = max_value
                        max_indices[k,c,a,b] = (h_start+max_row, w_start+max_col)
                        
        self.output = output
        requires_grad = x.requires_grad
            
        
        
        def grad_fn(grad):
            if x.requires_grad:
                maxpool_grad = np.zeros_like(self.input.data)
                for k in range(batch_size):
                    for c in range(channels):
                        for a in range(g):
                            for b in range(h):
                                
                                max_row, max_col = map(int,max_indices[k,c,a,b])
                                
                                maxpool_grad[k,c,max_row, max_col]+=grad[k,c,a,b]
                                
                self.input.backward(maxpool_grad)
        return Tensor(self.output, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None) 
    
class Conv2D:
    def __init__(self, in_features, out_features, stride, kernel_size, padding):
        scale = np.sqrt(2.0/in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        
        self.padding = padding
        self.stride = stride
        a, b = self.kernel_size[0], self.kernel_size[1]
        self.filters = Tensor(np.random.randn(out_features, in_features, a, b)*scale, requires_grad = True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        
    def __call__(self, x):
        self.x = x
        a, b = self.kernel_size[0], self.kernel_size[1]
        f, c, h, w = self.filters.data.shape
        batch_size = x.data.shape[0]
       
        if self.padding > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
                   
        else:
            x_padded = x.data
        
        padded_height , padded_width = x_padded.shape[2], x_padded.shape[3]
        
        output_height = (padded_height - h) // self.stride + 1
        output_width = (padded_width- w) // self.stride + 1
        
        
        
        output = np.zeros((x.data.shape[0], f, output_height, output_width))
        # print(output )
        for batch_idx in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    
                        h_start = h*self.stride
                        w_start = w*self.stride
                        
                        window = x_padded[batch_idx,:,h_start:h_start+a, w_start:w_start+b]
                    
                        for filter_idx in range(f):
                            output[batch_idx, filter_idx, h, w]= np.sum(window * self.filters.data[filter_idx]) + self.bias.data[filter_idx]
                            
        requires_grad = self.filters.requires_grad or self.bias.requires_grad or x.requires_grad
                    
        def grad_fn(grad):
            
            if self.filters.requires_grad:
                filter_grad = np.zeros_like(self.filters.data)

                for batch_idx in  range(batch_size):
                    for h in range(output_height):
                        for w in range(output_width):
                            h_start = h*self.stride
                            w_start = w*self.stride
                            window=self.x.data[batch_idx,:,h_start:h_start+a, w_start:w_start+b]
                            if ((h_start+a)>self.x.data.shape[2] or (w_start+b)>self.x.data.shape[3]):
                                continue
                            for filter_idx in range(f):
                                grad_value = (grad[batch_idx, filter_idx,h,w])
                                
                                filter_grad[filter_idx]+=window*grad_value
                    self.filters.grad = filter_grad
            if self.bias.requires_grad:
                self.bias.grad = np.sum(grad, axis =(0,2, 3))
            if x.requires_grad:
                input_grad = np.zeros_like(self.x.data)
                for batch_idx in range(batch_size):
                    for h in range(output_height):
                        for w in range(output_width):
                            h_start = h *self.stride
                            w_start = w*self.stride
                            window = self.x.data[batch_idx,:,h_start:h_start+a, w_start :w_start+b]
                            if ((h_start+a)>self.x.data.shape[2] or (w_start+b)>self.x.data.shape[3]):
                                continue
                            for filter_idx in range(f):
                                grad_value = float(grad[batch_idx,filter_idx,h,w])
                              
                                input_grad[batch_idx,:,h_start:h_start+a, w_start:w_start+b] +=self.filters.data[filter_idx]*grad_value 
                    self.x.backward(input_grad)
                
        return Tensor(output, requires_grad = requires_grad, grad_fn=grad_fn if requires_grad else None)
    
    
    def parameters(self):           
                            
        return[self.filters, self.bias]
                
