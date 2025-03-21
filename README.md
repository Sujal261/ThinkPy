# Neural Network Library (ANN)

## Overview
This project is a simple neural network library implemented in Python. It provides fundamental components for building, training, and evaluating artificial neural networks (ANNs). The library includes custom implementations of tensors, layers, activation functions, loss functions, and optimizers.

## Features
- Custom **Tensor** implementation to handle computations.
- **Linear layers** to build neural networks.
- Common **activation functions** such as ReLU.
- **Loss functions**, including Mean Squared Error (MSE).
- **Optimizers** for training the network.
- **Visualization utilities** for loss curves and dataset visualization.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Sujal261/ThinkPy.git
   cd ThinkPy
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Demo
The following demo showcases how to use this library to create and train a simple ANN for linear regression and classification.

There are two files demonstrating the usage of this library:
- **`ann.py`**: Implements a simple artificial neural network for regression.
- **`classification.py`**: Implements a classification model using the library.

### `ann.py` - Simple ANN for Regression
This example demonstrates how to use the library for training a simple ANN for linear regression.

### Import necessary modules
```python
import numpy as np 
import matplotlib.pyplot as plt
from src.loss_functions import MSELoss
from src.optimizers import Optimizer
from src.layers import Linear 
from src.Tensor import Tensor
from utils.plotting_curves import loss_curve
from src.Actication import ReLU
```

### Generate synthetic dataset
A simple linear relationship is defined as:
\[ y = 0.7x + 0.3 \]
We generate input values and split them into training and testing sets.
```python
weight = 0.7 
bias = 0.3
step = 0.02
 
start = 0
end = 1
x = np.arange(start, end, step ).reshape(-1,1)
y = weight*x + bias

# Splitting data
train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# Converting to Tensor objects
x_train_tensor = Tensor(x_train)
y_train_tensor = Tensor(y_train)
x_test_tensor = Tensor(x_test, requires_grad=False)
y_test_tensor = Tensor(y_test, requires_grad=False)
```

### Visualize dataset
```python
def plot_curve(train_data, train_label, test_data, test_label):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_label, c='b', label='Train Data')
    plt.scatter(test_data, test_label, c='g', label='Test Data')
    plt.legend()
    plt.show()

plot_curve(x_train, y_train, x_test, y_test)
```

### Define the neural network model
The model consists of:
- A **Linear** layer with 6 neurons.
- A **ReLU activation** function.
- A **Linear** output layer with 1 neuron.
```python
class SimpleModel:
    def __init__(self, in_features, out_features):
        self.layer1 = Linear(in_features=in_features, out_features=6)
        self.relu = ReLU()
        self.layer2 = Linear(in_features=6, out_features=out_features)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    
    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()

model = SimpleModel(1, 1)
```

### Define loss function and optimizer
```python
loss_fn = MSELoss()
optimizer = Optimizer(params=model.parameters(), lr=0.01)
```

### Train the model
```python
epochs = 200
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(x_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Evaluate on test set
    y_pred_test = model.forward(x_test_tensor)
    loss_test = loss_fn(y_pred_test, y_test_tensor)
    
    train_loss_values.append(loss.data)
    test_loss_values.append(loss_test.data)
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss.data:.4f} | Test Loss: {loss_test.data:.4f}")
```

### Plot results
```python
y_pred_test = model.forward(x_test_tensor)
plot_curve(x_train, y_train, x_test, y_pred_test.data)
loss_curve(train_loss_values, 'Train Loss')
loss_curve(test_loss_values, 'Test Loss')
```

### Explanation of Usage
- The **dataset** is generated based on a linear function.
- The data is **split** into training and test sets.
- The **SimpleModel** is defined with a two-layer architecture using `Linear` and `ReLU`.
- The **loss function** (MSELoss) and **optimizer** are initialized.
- The **training loop** runs for 200 epochs, performing forward propagation, loss computation, backpropagation, and optimization.
- The **loss curves** and predictions are plotted for analysis.

This example demonstrates how to use the library for training a simple ANN for regression.

For classification tasks, refer to **`classification.py`**, which implements a similar ANN architecture adapted for classification problems.


# Contributions
Feel free to contribute to the project by adding more features, optimizing training, or improving model architecture!

