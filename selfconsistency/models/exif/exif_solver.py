import os, sys, numpy as np, time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from selfconsistency.lib.utils import ops, io
import traceback
from collections import deque

class ExifSolver(object):
    def __init__(self, checkpoint=None, use_exif_summary=True, exp_name='no_name', init_summary=True):
        """
        Args
            checkpoint: .ckpt file to initialize weights from
            use_exif_summary: EXIF accuracy are stored
            exp_name: ckpt and tb name prefix
            init_summary: will create TB files, will override use_exif_summary arg
        """
        self.checkpoint = None if checkpoint in ['', None] else checkpoint
        self.exp_name = exp_name
        self._batch_size = 128
        self.use_exif_summary = use_exif_summary
        self.init_summary = init_summary
        self.ckpt_path = os.path.join('./ckpt', exp_name, exp_name)
        io.make_dir(self.ckpt_path)

        self.train_iterations = 10000000
        self.test_init = True
        self.show_iter = 20
        self.test_iter = 2000
        self.save_iter = 10000

        self.train_timer = deque(maxlen=10)
        return

    def setup_net(self, net):
        """ Links and setup loss and summary """
        # Link network
        self.net = net

        # Initialize some basic things
        self.sess = tf.Session(config=ops.config(self.net.use_gpu))
        if self.init_summary:
            self.train_writer = tf.summary.FileWriter(os.path.join('./tb', self.exp_name + '_train'), self.sess.graph)
            self.test_writer  = tf.summary.FileWriter(os.path.join('./tb', self.exp_name + '_test'))
            self.setup_summary()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        # Try to load checkpoint
        if self.checkpoint is not None:
            assert os.path.exists(self.checkpoint) or os.path.exists(self.checkpoint + '.index'), 'checkpoint does not exist'
            try:
                self.saver.restore(self.sess, self.checkpoint)
                self.i = io.parse_checkpoint(self.checkpoint)
                print('Succesfully resuming from %s' % self.checkpoint)
            except Exception:
                print(traceback.format_exc())
                try:
                    print('Model and checkpoint did not match, attempting to restore only weights')
                    variables_to_restore = ops.get_variables(self.checkpoint, exclude_scopes=['Adam'])
                    restorer = tf.train.Saver(variables_to_restore)
                    restorer.restore(self.sess, self.checkpoint)
                except Exception:
                    print('Model and checkpoint did not match, attempting to partially restore')
                    self.sess.run(tf.global_variables_initializer())
                    # Make sure you correctly set exclude_scopes if you are finetuining models or extending it
                    variables_to_restore = ops.get_variables(self.checkpoint, exclude_scopes=['classify']) #'resnet_v2_50/logits/', 'predict',
                    restorer = tf.train.Saver(variables_to_restore)
                    restorer.restore(self.sess, self.checkpoint)

                print('Variables intitializing from scratch')
                for var in tf.trainable_variables():
                    if var not in variables_to_restore:
                        print(var)
                print('Succesfully restored %i variables' % len(variables_to_restore))
                self.i = 0
        else:
            print('Initializing from scratch')
            self.i = 0
            self.sess.run(tf.global_variables_initializer())
        self.start_i = self.i

        if self.net.use_tf_threading:
            self.coord = tf.train.Coordinator()
            self.net.train_runner.start_p_threads(self.sess)
            tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        return


    def setup_data(self, data, data_fn=None):
        assert not self.net.use_tf_threading, "Using queue runner"
        self.data = data
        if data_fn is not None:
            self.data_fn = data_fn
        else:
            try:
                self.data_fn = self.data.exif_balanced_nextbatch
            except:
                self.data_fn = self.data.nextbatch

        assert self.data_fn is not None
        return

    def get_data(self, batch_size, split='train'):
        """ Make sure to pass None even if not using final classification """
        assert self.data is not None
        if batch_size is None:
            batch_size = self._batch_size

        data_dict = self.data_fn(batch_size, split=split)

        args = {
                self.net.im_b:data_dict['im_b']}

        if 'cls_lbl' in data_dict:
            args[self.net.cls_label] = data_dict['cls_lbl']

        if 'exif_lbl' in data_dict:
            args[self.net.label] = data_dict['exif_lbl']
        return args

    def train(self):
        print('Started training')
        while self.i < self.train_iterations:
            if self.test_init and self.i == self.start_i:
                print('Testing initialization')
                self.test(writer=self.test_writer)

            self._train()
            self.i += 1

            if self.i % self.show_iter == 0:
                self.show(writer=self.train_writer, phase='train')

            if self.i % self.test_iter == 0:
                self.test(writer=self.test_writer)

            if self.i % self.save_iter == 0 and self.i != self.start_i:
                io.make_ckpt(self.saver, self.sess, self.ckpt_path, self.i)
        return

    def _train(self):
        start_time = time.time()
        if self.net.use_tf_threading:
            self.sess.run(self.net.opt)
        else:
            args = self.get_data(self.net.batch_size, 'train')
            self.sess.run(self.net.opt, feed_dict=args)
        self.train_timer.append(time.time() - start_time)
        return

    def show(self, writer, phase='train'):
        if self.net.use_tf_threading:
            summary = self.sess.run(self.summary)
        else:
            args = self.get_data(self.net.batch_size, phase)
            summary = self.sess.run(self.summary, feed_dict=args)

        io.add_summary(writer, summary, self.i)

        io.show([['Train time', np.mean(list(self.train_timer))]],
                 phase=phase, iter=self.i)
        return

def initialize(args):
    return ExifSolver(**args)
