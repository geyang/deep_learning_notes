import tensorflow as tf


def get_shape_array(tensor):
    return to_array(tensor.get_shape())


def to_array(shape):
    return list(map(lambda d: d.value, list(shape)))


"""deprecated: now you can do tf.one_hot"""
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
