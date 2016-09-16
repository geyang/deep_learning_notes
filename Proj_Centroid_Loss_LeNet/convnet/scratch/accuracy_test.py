import tensorflow as tf

output = [
    [0, 0, 0, 0, 0],
    [0, 0, 1.0, 0, 0],
    [1.0, 0, 0, 0, 0]
]
batch_labels = [
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0]
]

with tf.Graph().as_default(), tf.Session() as sess:
    logits = tf.constant(output, dtype=tf.float32)
    labels = tf.constant(batch_labels, dtype=tf.int32)
    correct = tf.nn.in_top_k(logits, tf.cast(tf.argmax(labels, dimension=1), dtype=tf.int32), 1)

    accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))

    list_correct, count_correct = sess.run([correct, accuracy])
    assert list_correct[0] == False, 'empty array should be false'
    assert list_correct[1] == False, 'wrong logits should be false'
    assert list_correct[2] == True, 'should be correct'
    print(list_correct, count_correct)
