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


if __name__ == '__main__':
    with tf.Session() as sess:
        labels = one_hot_encoding_batch([0, 1, 2, 3, 4], 5, 10)
        print('label one hot dimension', labels.get_shape())
        output = sess.run(labels)

        assert output[0, 0], 'First label should be 1 (0)'
        assert output[1, 1], 'Second label should be 1 (1)'
        assert output[2, 2], 'Third label should be 1 (2)'
        assert output[3, 3], 'Fourth label should be 1 (3)'

        # Now, tensorflow has the `tf.one_hot` function which does the same thing (for the most part)
        # instead of above, you can:

        labels = tf.one_hot(
            indices=[0, 2, -1, 1],
            depth=3,
            on_value=5.0,
            off_value=0.0,
            axis=-1
        )
        output = sess.run(labels)
        print(output)
        # output =
        # [5.0 0.0 0.0]  // one_hot(0)
        # [0.0 0.0 5.0]  // one_hot(2)
        # [0.0 0.0 0.0]  // one_hot(-1)
        # [0.0 5.0 0.0]  // one_hot(1)

        assert output[0, 0], 'First label should be 1 (0)'
        assert output[1, 1], 'Second label should be 1 (1)'
        assert output[2, 2], 'Third label should be 1 (2)'
        assert output[3, 3], 'Fourth label should be 1 (3)'
