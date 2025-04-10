import math
import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

# Calculate exponentials
exp_values = np.exp(layer_outputs)
print(np.sum(layer_outputs, axis=0, keepdims=True))

norm_values = exp_values / np.sum(exp_values, axis=1,keepdims=True)
print(norm_values)

# Normalize by dividing by the sum of exponentials
# norm_values = exp_values / np.sum(exp_values)

# Output the normalized values and their sum

#
# print("Softmax Output:", norm_values)
# print("Sum of Softmax Output:", np.sum(norm_values))  # Should be ~1.0
