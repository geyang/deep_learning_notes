import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
    """ prints:
    Device mapping:
    /job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: GRID K520, pci bus id: 0000:00:03.0
    /job:localhost/replica:0/task:0/gpu:1 -> device: 1, name: GRID K520, pci bus id: 0000:00:04.0
    /job:localhost/replica:0/task:0/gpu:2 -> device: 2, name: GRID K520, pci bus id: 0000:00:05.0
    /job:localhost/replica:0/task:0/gpu:3 -> device: 3, name: GRID K520, pci bus id: 0000:00:06.0
    I tensorflow/core/common_runtime/direct_session.cc:175] Device mapping:
    /job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: GRID K520, pci bus id: 0000:00:03.0
    /job:localhost/replica:0/task:0/gpu:1 -> device: 1, name: GRID K520, pci bus id: 0000:00:04.0
    /job:localhost/replica:0/task:0/gpu:2 -> device: 2, name: GRID K520, pci bus id: 0000:00:05.0
    /job:localhost/replica:0/task:0/gpu:3 -> device: 3, name: GRID K520, pci bus id: 0000:00:06.0

    MatMul: /job:localhost/replica:0/task:0/gpu:0
    I tensorflow/core/common_runtime/simple_placer.cc:818] MatMul: /job:localhost/replica:0/task:0/gpu:0
    b: /job:localhost/replica:0/task:0/gpu:0
    I tensorflow/core/common_runtime/simple_placer.cc:818] b: /job:localhost/replica:0/task:0/gpu:0
    a: /job:localhost/replica:0/task:0/gpu:0
    I tensorflow/core/common_runtime/simple_placer.cc:818] a: /job:localhost/replica:0/task:0/gpu:0
    [[ 22.  28.]
     [ 49.  64.]]
    """