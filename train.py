import tensorflow as tf
from tqdm import tqdm

from config import cfg
from capsNet import CapsNet


if __name__ == "__main__":
    capsNet = CapsNet(is_training=cfg.is_training)
    tf.logging.info('Graph loaded')
    sv = tf.train.Supervisor(graph=capsNet.graph,
                             logdir=cfg.logdir,
                             save_model_secs=0)

    with sv.managed_session() as sess:
        num_batch = int(60000 / cfg.batch_size)
        for epoch in range(cfg.epoch):
            if sv.should_stop():
                break
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                sess.run(capsNet.train_op)

            global_step = sess.run(capsNet.global_step)
            sv.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

    tf.logging.info('Training done')
