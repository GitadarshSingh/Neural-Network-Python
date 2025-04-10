inputs = [1, 2, 3, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]

some_value = 0.5
weight = -0.7
bias = 0.7

print(some_value*weight)
print(some_value+bias)

'''
layer_outputs = []  # Output for current layer

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0  # Output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight  # Multiply inputs × weights
    neuron_output += neuron_bias  # Add bias after all inputs processed
    layer_outputs.append(neuron_output)  # Store output

print(layer_outputs)
'''