import reverb
import time
from configparser import ConfigParser
import sys
import os
from tqdm import tqdm
from multiprocessing import Pool

conf_file = sys.argv[1]
# conf_file = 'base_example/baseline/exp_1.ini'
config = ConfigParser()
config.read(conf_file)

os.environ['CUDA_VISIBLE_DEVICES'] = config.get('public', 'CUDA_VISIBLE_DEVICES')

CLIENT_NUM_LIST = list(map(int, config.get('client', 'CLIENT_NUM').split(',')))

TEST_EPISODE_NUM = config.getint('client', 'TEST_EPISODE_NUM')
BATCH_SIZE = config.getint('client', 'test_batch_size')

server_addr = config.get('public', 'server_addr') + ':' + config.get('public', 'port')
# remote_client = reverb.Client(server_addr)
# print(remote_client.server_info())

table_name_list = ['Uniform_table', 'Prioritized_table', 'MinHeap_table', 'MaxHeap_table']

def single_process(idx):
    print(f"Client: {str(idx)} starts: ")
    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=server_addr,
        table='Test_table',
        max_in_flight_samples_per_worker=3*BATCH_SIZE,
        rate_limiter_timeout_ms=10)

    temp_dataset = dataset.batch(BATCH_SIZE)
    start_time = time.time()
    for _ in range(TEST_EPISODE_NUM):
        # pbar.update(1)
        sample = next(iter(temp_dataset))
    cost_time = time.time() - start_time
    throughput = TEST_EPISODE_NUM * BATCH_SIZE / cost_time
    # str_ = f'Batch Size: {BATCH_SIZE}, Throughput: {throughput} samples/s'
    # print(str_)
    return throughput
        
start_time = time.time()
fo = open(f"./result/{conf_file}_res.txt", "w")
config.write(fo)
for client_num in CLIENT_NUM_LIST:
    proc_list = []
    print(f"Start to test {client_num} clients")
    proc_pool = Pool(client_num)
    for idx in range(client_num):
        proc_list.append(proc_pool.apply_async(single_process, args=(idx,)))
    proc_pool.close()
    proc_pool.join()
    res = 0
    fo.write(f"\nClient Num: {client_num}\n")
    fo.flush()
    for idx, proc in enumerate(proc_list):
        tmp = proc.get()
        res += tmp
        fo.write(f" Client {idx} throughput: {tmp} samples/s\n")
    fo.write(f" Total throughput: {res*0.46875/1024} GB/s, {res} samples/s; Time Cost: {time.time()-start_time}\n")
    fo.flush()
fo.close()