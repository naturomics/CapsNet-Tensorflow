import sys

sys.path.append('.')

import tensorflow as tf
from config import cfg
from utils import load_mnist
import dist_version.capsnet_slim as net
import time
import tensorflow.contrib.slim as slim
import re
import copy
import numpy as np
import os

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

def tower_loss(x, y, scope, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        v_len, output = net.build_arch(x, y, is_train=True)

    net.loss(v_len, output, x, y)

    loss = tf.get_collection(tf.GraphKeys.LOSSES, scope)[0]
    loss_name = re.sub('%s_[0-9]*/' % 'tower_', '', loss.op.name)
    tf.summary.scalar(loss_name, loss)

    return loss

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def main(_):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

        num_batches_per_epoch = int(60000/(cfg.batch_size_per_gpu*cfg.num_gpu))

        opt = tf.train.AdamOptimizer()

        batch_x, batch_labels = create_inputs()
        batch_y = tf.one_hot(batch_labels, depth=10, axis=1, dtype=tf.float32)
        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

        x_splits = tf.split(axis=0, num_or_size_splits=cfg.num_gpu, value=batch_x)
        y_splits = tf.split(axis=0, num_or_size_splits=cfg.num_gpu, value=batch_y)

        tower_grads = []
        reuse_variables = None
        for i in range(cfg.num_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('tower_', i)) as scope:
                    with slim.arg_scope([slim.variable], device='/cpu:0'):
                        loss = tower_loss(x_splits[i], y_splits[i], scope, reuse_variables)

                    reuse_variables = True

                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

        grad = average_gradients(tower_grads)

        summaries.extend(input_summaries)

        train_op = opt.apply_gradients(grad, global_step=global_step)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfg.epoch)
        summary_op = tf.summary.merge(summaries)
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(
            cfg.logdir,
            graph=sess.graph)

        for step in range(cfg.epoch*num_batches_per_epoch):
            tic = time.time()
            _, loss_value = sess.run([train_op, loss])
            print(str(time.time()-tic)+' '+str(step))

            assert not np.isnan(loss_value)

            if step % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % num_batches_per_epoch == 0 or (step+1) == cfg.epoch*num_batches_per_epoch:
                ckpt_path = os.path.join(cfg.logdir, 'model.ckpt')
                saver.save(sess, ckpt_path, global_step=step)

if __name__ == "__main__":
    tf.app.run()
