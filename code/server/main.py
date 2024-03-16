# import collections
#
# import numpy as np
# import tensorflow as tf
# import tensorflow_federated as tff
#
# np.random.seed(0)
#
# print(tff.federated_computation(lambda: 'Hello, World!')())

# import tensorflow as tf
#
# print(f'\nTensorflow version = {tf.__version__}\n')
# print(f'\n{tf.config.list_physical_devices("GPU")}\n')

import torch

print(f'\nAvailable cuda = {torch.cuda.is_available()}')
print(f'\nGPUs availables = {torch.cuda.device_count()}')
print(f'\nCurrent device = {torch.cuda.current_device()}')
print(f'\nCurrent Device location = {torch.cuda.device(0)}')
print(f'\nName of the device = {torch.cuda.get_device_name(0)}')