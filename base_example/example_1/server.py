import reverb
import tensorflow as tf

EPISODE_LENGTH = 100

action_spec = tf.TensorSpec([1], dtype=tf.int32)
obsversation_spec = tf.TensorSpec([4, 84, 84], dtype=tf.float32)

server = reverb.Server(tables=[
    reverb.Table(
        name='test_table',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature={
            'actions':
                tf.TensorSpec([EPISODE_LENGTH, *action_spec.shape], action_spec.dtype),
            'observations':
                tf.TensorSpec([EPISODE_LENGTH, *obsversation_spec.shape], obsversation_spec.dtype),
        },
    )],
    port=52023
)
local_client = server.localhost_client()
print(local_client.server_info())