#using Dot Products

import numpy as np

inputs = [1, 2, 3, 2.5]
#weights = [0.2, 0.8, -0.5, 1.0] # 4.8
# bias = 2
# output = np.dot(inputs,weights)+ bias
# print(output)  4.8

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]


# output = np.dot(inputs,weights)+ biases
# print(output)   : Error
# Traceback (most recent call last):
#   File "C:\Users\ADARSH SINGH\movie-recommended-system\NeuralNetworksProject\NeuralNetworks\p3(a).py", line 20, in <module>
#     output = np.dot(inputs,weights)+ biases
#              ^^^^^^^^^^^^^^^^^^^^^^
# ValueError: shapes (4,) and (3,4) not aligned: 4 (dim 0) != 3 (dim 0)


# to fix the above error



output = np.dot(weights, inputs) + biases
print(output)