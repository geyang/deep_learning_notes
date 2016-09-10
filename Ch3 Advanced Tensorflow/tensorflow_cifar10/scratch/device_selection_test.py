import tensorflow as tf

"""
Instruction to the code there can be found at:
https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
"""

# Creates a graph.
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

session_config = tf.ConfigProto(
    log_device_placement=True
)
# we can either use gradual memory allocation, *OR* use a different gpu.
session_config.gpu_options.allow_growth = True
session_config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=session_config) as sess:
    result = sess.run(c)
    print(result)
