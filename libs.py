import tensorflow as tf

def conv_layer(input, in_channels, num_outputs,
               kernel_size, stride, padding, act=tf.nn.relu):
    W = tf.get_variable('W',
        initializer=tf.truncated_normal([kernel_size, kernel_size,
            in_channels, num_outputs], stddev=0.1))
    conv = act(tf.nn.conv2d(input, W,
        strides=[1, stride, stride, 1], padding=padding))

    tf.summary.histogram('W', W)
    tf.summary.histogram('conv', conv)

    return conv
