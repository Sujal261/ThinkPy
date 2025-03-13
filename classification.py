import numpy as np 
import matplotlib.pyplot as plt
from src.loss_functions import MSELoss, BinaryCrossEntropyLoss
from src.optimizers import Optimizer
from src.layers import Linear 
from src.Tensor import Tensor
from src.Actication import ReLU, Sigmoid
from utils.plotting_curves import loss_curve
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=100, noise = 0.1, factor = 0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    

X_train_tensor = Tensor(X_train, requires_grad=True)
y_train_tensor = Tensor(y_train.reshape(-1,1), requires_grad=True)



# Fix the model class
class Classification:
    def __init__(self, in_features, out_features):
        self.linear_layer1 = Linear(in_features, 8)
        self.relu1 = ReLU()
        self.linear_layer2 = Linear(8, 8)
        self.relu2 = ReLU()
        self.linear_layer3= Linear(8, out_features)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        x = self.linear_layer1(x)
        x = self.relu1(x)
        x = self.linear_layer2(x)
        x = self.relu2(x)
        x = self.linear_layer3(x)
        x = self.sigmoid(x)
        return x
    
    def parameters(self):
        return self.linear_layer1.parameters() + self.linear_layer2.parameters()+self.linear_layer3.parameters()

model = Classification(2, 1)
loss_fn = BinaryCrossEntropyLoss()
optimizer = Optimizer(model.parameters(), lr=0.1)

losses = []
epochs = 500
for epoch in range(epochs):
   
    y_pred = model.forward(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    losses.append(loss.data)
    
    
    optimizer.zero_grad()
    loss.backward()  
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.data:.4f}")