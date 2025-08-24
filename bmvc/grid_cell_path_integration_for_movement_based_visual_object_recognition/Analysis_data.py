# To inspect data in "training_and_testing_data1, 2"

import os
import numpy as np

x = np.load("python2_htm_docker/docker_dir/training_and_testing_data2/mnist_SDRs_base_net_training.npy")
# x.spahe = (54000, 6272), 6272 / 49 = 128
# x = data x (patch x dim)

y = np.load("python2_htm_docker/docker_dir/training_and_testing_data2/mnist_labels_base_net_training.npy")
# y.shape = (54000,)
print("y", y[0:10])

'''
x_samples = []
for mnist_iter in range(1):
        indices = np.nonzero(y == mnist_iter)
        print("indices", indices)
        # Get num_samples_per_class of each digit/class type
        x_samples.extend(x[indices][0: 1]) #x[indices].shape = (5400, 6272)
        print("x_samples", len(x_samples))
'''

x0 = x[0].reshape(128, 7, 7)
x0_0 = x0[:, 0, 0]
print(x0_0)
idx = np.where(x0_0 > 0)[0]
print("idx", idx)