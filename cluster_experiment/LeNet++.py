import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# parametric rectified linear unit
def prelu(x, alphas):
    return tf.nn.relu(x) + tf.mul(alphas, tf.sub(x, tf.abs(x)))

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
alphas_conv1 = bias_variable([32])
layer_conv_1 = prelu(conv2d(x_image, W_conv1) + b_conv1, alphas_conv1)

W_conv2 = weight_variable([5, 5, 32, 1])
b_conv2 = bias_variable([32])
alphas_conv2 = bias_variable([32])
layer_conv_2 = prelu(conv2d(layer_conv_1, W_conv2) + b_conv2, alphas_conv2)
stage_1_pool = max_pool_2x2(layer_conv_2)

layer_conv_2.get_shape()

W_conv3 = weight_variable([5, 5, 32, 64])
b_conv3 = bias_variable([64])
alphas_conv3 = bias_variable([64])
layer_conv_3 = prelu(conv2d(layer_conv_2, W_conv3) + b_conv3, alphas_conv3)

W_conv4 = weight_variable([5, 5, 64, 1])
b_conv4 = bias_variable([64])
alphas_conv4 = bias_variable([64])

layer_conv_4 = prelu(conv2d(layer_conv_3, W_conv4) + b_conv4, alphas_conv4)
stage_2_pool = max_pool_2x2(layer_conv_4)

## Stage 3:

W_conv5 = weight_variable([5, 5, 64, 128])
b_conv5 = bias_variable([128])
alphas_conv5 = bias_variable([128])
layer_conv_5 = prelu(conv2d(layer_conv_4, W_conv5) + b_conv5, alphas_conv5)

W_conv6 = weight_variable([5, 5, 128, 1])
b_conv6 = bias_variable([128])
alphas_conv6 = bias_variable([128])
layer_conv_6 = prelu(conv2d(layer_conv_5, W_conv6) + b_conv6, alphas_conv6)

stage_3_pool = max_pool_2x2(layer_conv_6)

stage_3_pool.get_shape()

W_fc1 = weight_variable([14 * 14 * 128, 2])
b_fc1 = bias_variable([2])
alphas_fc1 = bias_variable([2])

stage_3_pool_flat = tf.reshape(stage_3_pool, [-1, 14 * 14 * 128])
h_fc1 = prelu(tf.matmul(stage_3_pool_flat, W_fc1) + b_fc1, alphas_fc1)

# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# W_fc2 = weight_variable([2, 10])
# b_fc2 = bias_variable([10])

# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

W_fc2 = weight_variable([2, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(200000):
  batch = mnist.train.next_batch(250)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1]}) # , keep_prob: 1.0
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})#, keep_proj
