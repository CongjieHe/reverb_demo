# test torch tensor to tensorflow tensor

import tensorflow as tf
import numpy as np
import time
import os
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# test_epoch = 1000

# if __name__ == '__main__':
#     random_list = [torch.rand((100, 4, 84, 84), dtype=torch.float32) for i in range(test_epoch)]
#     time_1 = time.time()
#     for i in range(test_epoch):
#         tmp = random_list[i].numpy()
#         tf_tensor = tf.convert_to_tensor(tmp)
#     time_1 = time.time() - time_1    
    
#     random_list = [tf.random.normal((100, 4, 84, 84), dtype=tf.float32) for i in range(test_epoch)]
#     time_2 = time.time()
#     for i in range(test_epoch):
#         tmp = random_list[i].numpy()
#         torch_tensor = torch.from_numpy(tmp)
#     time_2 = time.time() - time_2
    
#     print('pytorch to tf time: ', time_1 / test_epoch)
#     print('tf to pytorch time: ', time_2 / test_epoch)

actions_spec = tf.TensorSpec([3, 1], dtype=tf.int32)
state_spec = tf.TensorSpec([3, 38], dtype=tf.float32)
rewards_spec = tf.TensorSpec([3, 1], dtype=tf.float32)
pbar = tqdm(total=1000*10000)
pbar.set_description("Processing: ")
start_time = time.time()
for _ in range(1000*10000):
    pbar.update(1)
    action = tf.random.uniform(actions_spec.shape, maxval=3, dtype=actions_spec.dtype)
    reward = tf.random.uniform(rewards_spec.shape, maxval=1, dtype=rewards_spec.dtype)
    state = tf.random.uniform(state_spec.shape, maxval=1, dtype=state_spec.dtype)
cost_time = time.time() - start_time
print(f"Cost time: {cost_time}")