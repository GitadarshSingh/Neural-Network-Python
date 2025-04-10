import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]


# Generate spiral dataset
X, y = spiral_data(100, 3)

# Dense layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation class
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Create layers
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

# layer2 = Layer_Dense(5, 2)
# activation2 = Activation_ReLU()

# Forward pass
layer1.forward(X)

activation1.forward(layer1.output)

# layer2.forward(activation1.output)
# activation2.forward(layer2.output)

# Output
print(activation1.output)  # Show only first 5 rows for readability
