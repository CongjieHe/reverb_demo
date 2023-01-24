import reverb
import tensorflow as tf
from tqdm import tqdm
import random
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ['0']

NUM_EPISODES = 20000
EPISODE_LENGTH = 1000

actions_spec = tf.TensorSpec([3, 1], dtype=tf.int32)
state_spec = tf.TensorSpec([3, 38], dtype=tf.float32)
rewards_spec = tf.TensorSpec([3, 1], dtype=tf.float32)

server = reverb.Server(tables=[
    reverb.Table(
        name='Uniform_table',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=NUM_EPISODES,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature={
            'actions':
                tf.TensorSpec([EPISODE_LENGTH, *actions_spec.shape], actions_spec.dtype),
            'rewards':
                tf.TensorSpec([EPISODE_LENGTH, *rewards_spec.shape], rewards_spec.dtype),
            'states':
                tf.TensorSpec([EPISODE_LENGTH, *state_spec.shape], state_spec.dtype),
        },
    ),
    reverb.Table(
        name='Prioritized_table',
        sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
        remover=reverb.selectors.Fifo(),
        max_size=NUM_EPISODES,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature={
            'actions':
                tf.TensorSpec([EPISODE_LENGTH, *actions_spec.shape], actions_spec.dtype),
            'rewards':
                tf.TensorSpec([EPISODE_LENGTH, *rewards_spec.shape], rewards_spec.dtype),
            'states':
                tf.TensorSpec([EPISODE_LENGTH, *state_spec.shape], state_spec.dtype),
        },
    ),
    reverb.Table(
        name='MinHeap_table',
        sampler=reverb.selectors.MinHeap(),
        remover=reverb.selectors.Fifo(),
        max_size=NUM_EPISODES,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature={
            'actions':
                tf.TensorSpec([EPISODE_LENGTH, *actions_spec.shape], actions_spec.dtype),
            'rewards':
                tf.TensorSpec([EPISODE_LENGTH, *rewards_spec.shape], rewards_spec.dtype),
            'states':
                tf.TensorSpec([EPISODE_LENGTH, *state_spec.shape], state_spec.dtype),
        },
    ),
    reverb.Table(
        name='MaxHeap_table',
        sampler=reverb.selectors.MaxHeap(),
        remover=reverb.selectors.Fifo(),
        max_size=NUM_EPISODES,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature={
            'actions':
                tf.TensorSpec([EPISODE_LENGTH, *actions_spec.shape], actions_spec.dtype),
            'rewards':
                tf.TensorSpec([EPISODE_LENGTH, *rewards_spec.shape], rewards_spec.dtype),
            'states':
                tf.TensorSpec([EPISODE_LENGTH, *state_spec.shape], state_spec.dtype),
        },
    )],
    port=52023
)

local_client = server.localhost_client()
print(local_client.server_info())

table_name_list = ['Uniform_table', 'Prioritized_table', 'MinHeap_table', 'MaxHeap_table']

with local_client.trajectory_writer(num_keep_alive_refs=EPISODE_LENGTH) as writer:
    action = tf.random.uniform(actions_spec.shape, maxval=3, dtype=actions_spec.dtype)
    reward = tf.random.uniform(rewards_spec.shape, maxval=1, dtype=rewards_spec.dtype)
    state = tf.random.uniform(state_spec.shape, maxval=1, dtype=state_spec.dtype)
    
    pbar = tqdm(total=NUM_EPISODES)
    pbar.set_description('Generating Fake Data:')
    for _ in range(NUM_EPISODES):
        pbar.update(1)
        for _ in range(EPISODE_LENGTH):
            # action = tf.random.uniform(actions_spec.shape, maxval=3, dtype=actions_spec.dtype)
            # reward = tf.random.uniform(rewards_spec.shape, maxval=1, dtype=rewards_spec.dtype)
            # state = tf.random.uniform(state_spec.shape, maxval=1, dtype=state_spec.dtype)
            
            writer.append({'action':action, 'reward':reward, 'state':state})
            
        for table_name in table_name_list:
            
            writer.create_item(
                table=table_name,
                priority=float(random.random()),
                trajectory={
                'actions': writer.history['action'][:],
                'rewards': writer.history['reward'][:],
                'states': writer.history['state'][:],
            })
        
        writer.end_episode(timeout_ms=1000)