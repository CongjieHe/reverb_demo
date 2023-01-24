# test torch tensor to tensorflow tensor

import torch
import tensorflow as tf
import numpy as np
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

test_epoch = 1000

if __name__ == '__main__':
    random_list = [torch.rand((100, 4, 84, 84), dtype=torch.float32) for i in range(test_epoch)]
    time_1 = time.time()
    for i in range(test_epoch):
        tmp = random_list[i].numpy()
        tf_tensor = tf.convert_to_tensor(tmp)
    time_1 = time.time() - time_1    
    
    random_list = [tf.random.normal((100, 4, 84, 84), dtype=tf.float32) for i in range(test_epoch)]
    time_2 = time.time()
    for i in range(test_epoch):
        tmp = random_list[i].numpy()
        torch_tensor = torch.from_numpy(tmp)
    time_2 = time.time() - time_2
    
    print('pytorch to tf time: ', time_1 / test_epoch)
    print('tf to pytorch time: ', time_2 / test_epoch)