import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

# Categorical cross-entropy loss
correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
loss = -np.log(correct_confidences)
mean_loss = np.mean(loss)

print(mean_loss)
