import sys

sys.path.append('.')

import tensorflow as tf
from config import cfg
from utils import load_mnist
import dist_version.capsnet_slim as net
import time
import tensorflow.contrib.slim as slim

def create_inputs():
    trX, trY = load_mnist(cfg.dataset, cfg.is_training)

    num_pre_threads = cfg.thread_per_gpu*cfg.num_gpu
    data_queue = tf.train.slice_input_producer([trX, trY], capacity=64*num_pre_threads)
    X, Y = tf.train.shuffle_batch(data_queue, num_threads=num_pre_threads,
                                  batch_size=cfg.batch_size_per_gpu*cfg.num_gpu,
                                  capacity=cfg.batch_size_per_gpu*cfg.num_gpu * 64,
                                  min_after_dequeue=cfg.batch_size_per_gpu*cfg.num_gpu * 32,
                                  allow_smaller_final_batch=False)

    return (X, Y)

def main(_):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

        num_batches_per_epoch = int(60000/(cfg.batch_size_per_gpu*cfg.num_gpu))

        opt = tf.train.AdamOptimizer()

        batch_x, batch_labels = create_inputs()
        batch_y = tf.one_hot(batch_labels, depth=10, axis=1, dtype=tf.float32)
        x_splits = tf.split(axis=0, num_or_size_splits=cfg.num_gpu, value=batch_x)
        y_splits = tf.split(axis=0, num_or_size_splits=cfg.num_gpu, value=batch_y)

        tower_grads = []
        reuse_variables = None
        for i in range(cfg.num_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('tower_', i)) as scope:
                    with slim.arg_scope([slim.variable], device='/cpu:0'):
                        output = net.build_arch(x_splits[i])
                        # loss = tower_loss(x_splits[i], y_splits[i], scope)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        tic = time.time()
        for i in range(100):
            o = sess.run(output)

        print(time.time()-tic)


if __name__ == "__main__":
    tf.app.run()
