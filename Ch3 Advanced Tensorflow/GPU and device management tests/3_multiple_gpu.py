import tensorflow as tf

# Creates a graph.
# Here we assign multiple matrix multiplications to multiple GPU's
c = []
for d in ['/gpu:2', '/gpu:3']:
    with tf.device(d):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
        c.append(tf.matmul(a, b))

# then use the CPU to add the result up together.
with tf.device('/cpu:0'):
    sum = tf.add_n(c)

# Creates a session with log_device_placement set to True.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(sum))
