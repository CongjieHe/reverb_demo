import reverb
import time
from configparser import ConfigParser
import sys
import os

conf_file = sys.argv[1]
config = ConfigParser()
config.read(conf_file)

os.environ['CUDA_VISIBLE_DEVICES'] = config.get('public', 'CUDA_VISIBLE_DEVICES')

test_batch_size = list(map(int, config.get('client', 'test_batch_size').split(',')))

TEST_EPISODE_NUM = config.getint('client', 'TEST_EPISODE_NUM')

server_addr = config.get('public', 'server_addr') + ':' + config.get('public', 'port')
remote_client = reverb.Client(server_addr)
print(remote_client.server_info())

table_name_list = ['Uniform_table', 'Prioritized_table', 'MinHeap_table', 'MaxHeap_table']
fo = open(f"./result/{conf_file}_res.txt", "w")
config.write(fo)
for table_name in table_name_list:
    str_ = f'\nTable Name: {table_name}'
    print(str_)
    fo.write(str_ + '\n')
    for batch_size in test_batch_size:
        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=server_addr,
            table=table_name,
            max_in_flight_samples_per_worker=3*batch_size,
            rate_limiter_timeout_ms=10)
    
        temp_dataset = dataset.batch(batch_size)
        start_time = time.time()
        for _ in range(TEST_EPISODE_NUM):
            sample = temp_dataset.take(1)
        cost_time = time.time() - start_time
        throughput = TEST_EPISODE_NUM * batch_size / cost_time
        str_ = f'Batch Size: {batch_size}, Throughput: {throughput} samples/s'
        print(str_)
        fo.write(str_ + '\n')
fo.close()