import reverb
import time
from tqdm import tqdm

# test_batch_size = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512]
test_batch_size = [8, 16, 24, 32, 40, 48, 56, 64]

TEST_EPISODE_NUM = 100

server_addr = 'localhost:52023'
remote_client = reverb.Client(server_addr)
print(remote_client.server_info())

table_name_list = ['Uniform_table', 'Prioritized_table', 'MinHeap_table', 'MaxHeap_table']

for table_name in table_name_list:
    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=server_addr,
        table=table_name,
        max_in_flight_samples_per_worker=10,
        rate_limiter_timeout_ms=10)
    
    print(f'\nTable Name: {table_name}')
    for batch_size in test_batch_size:
        temp_dataset = dataset.batch(batch_size)
        start_time = time.time()
        for _ in range(TEST_EPISODE_NUM):
            sample = temp_dataset.take(1)
        cost_time = time.time() - start_time
        throughput = TEST_EPISODE_NUM * batch_size / cost_time
        print(f'Batch Size: {batch_size}, Throughput: {throughput} samples/s')