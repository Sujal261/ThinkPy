import numpy as np 
import matplotlib.pyplot as plt
from src.loss_functions import MSELoss
from src.optimizers import Optimizer
from src.layers import Linear 
from src.Tensor import Tensor
from utils.plotting_curves import loss_curve
from src.Actication import ReLU
#Making data ready
weight = 0.7 
bias = 0.3
step = 0.02
 
start = 0
end = 1
x = np.arange(start, end, step )
x = x.reshape(-1,1)
y = weight*x+bias
#Making training and testing split
train_split = int(0.8*len(x))
test_split = len(x) - train_split
x_train , y_train = x[:train_split], y[:train_split]
x_test, y_test =x[train_split:], y[train_split:]
x_train_tensor = Tensor(x_train)
y_train_tensor = Tensor(y_train)
x_test_tensor = Tensor(x_test, requires_grad= False)
y_test_tensor = Tensor(y_test, requires_grad=False)
    
def plot_curve(train_data = x_train,
         train_label =y_train,
         test_data=x_test,
         test_label = y_test):
    
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_label, c='b')
    plt.scatter(test_data, test_label, c= 'g')
    plt.show()
    
plot_curve()
class simple():
    def __init__(self, in_features, out_features):
        self.layer1 = Linear(in_features=in_features,out_features=6)
        self.relu = ReLU()
        self.layer2 = Linear(in_features=6, out_features=out_features)
    def forward(self,x):
        x= self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    def parameters(self):
        return self.layer1.parameters()+self.layer2.parameters()
    
model1 = simple(1,1)


loss_fn = MSELoss()
optimizer = Optimizer(params= model1.parameters(), lr = 0.01)


epochs = 200
train_loss_values=[]
test_loss_values=[]


for epoch in range(epochs):
    y_pred = model1.forward(x_train_tensor)
    loss= loss_fn(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    y_pred_test = model1.forward(x_test_tensor)
    loss_test = loss_fn(y_pred_test, y_test_tensor)
    train_loss_values.append(loss.data)
    test_loss_values.append(loss_test.data)
    if epoch % 10 == 0:
        print(f"Epoch:{epoch}  train loss:{loss.data} test loss:{loss_test.data}")
        
y = model1.forward(x_test_tensor)
plot_curve(test_data = x_test,
     test_label =y.data)
loss_curve(train_loss_values, 'Train Loss')
loss_curve(test_loss_values,'Test Loss')
