import tensorflow as tf

output = [
    [0, 0, 0, 0, 0],
    [0, 1.0, 0, 0, 0],
    [1.0, 0, 0, 0, 0]
]
batch_labels = [
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0]
]

with tf.Graph().as_default(), tf.Session() as sess:
    logits = tf.constant(output, dtype=tf.float32)
    labels = tf.constant(batch_labels, dtype=tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits,
        tf.cast(tf.argmax(labels, dimension=1), dtype=tf.int64),
        name='xentropy'
    )

    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    result = sess.run([logits, cross_entropy, loss])
    print(result)
