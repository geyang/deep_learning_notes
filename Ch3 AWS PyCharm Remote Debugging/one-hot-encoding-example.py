import tensorflow as tf


def one_hot_encoding_batch(batch, batch_size, num_labels):
    # this transposes the labels. Sparse as the opposite of `dense`ly encoded later.
    sparse_labels = tf.reshape(batch, [batch_size, 1])

    # tf.range is similar to range() in python. It creates a list.
    # this creates a list of indices [0, batch_size),
    # and then transposes it.
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])

    # this is effectively a `zip` operation. Dimension(conca...) = [5, 2] in the example.
    concatenated = tf.concat(1, [indices, sparse_labels])

    # following two lines just just build a tuple of (batch_size, num_labels).
    output_shape = tf.concat(0, [[batch_size], [num_labels]])

    sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
    return tf.reshape(sparse_to_dense, [batch_size, num_labels])


with tf.Session() as sess:
    labels = one_hot_encoding_batch([0, 1, 2, 3, 4], 5, 10)
    print('label one hot dimension', labels.get_shape())
    result = sess.run(labels)

    assert result[0, 0], 'First label should be 1 (0)'
    assert result[1, 1], 'Second label should be 1 (1)'
    assert result[2, 2], 'Third label should be 1 (2)'
    assert result[3, 3], 'Fourth label should be 1 (3)'


    batch_size = 2
    num_labels = 3
    concat = tf.concat(0, [[batch_size], [num_labels]])
    output_shape = tf.reshape(concat, [2])
    result = sess.run(output_shape)
    print(result)
