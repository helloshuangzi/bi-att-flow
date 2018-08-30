import gzip
import json
from json import encoder
import os

import tensorflow as tf

from basic.evaluator import Evaluation, F1Evaluation
from my.utils import short_floats

import pickle
import my.hyperdrive_utils as hyperdrive_utils

class GraphHandler(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        self.writer = None
        self.save_path = os.path.join(config.save_dir, config.model_name)

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
        if self.config.load:
            self._load(sess)
        elif hyperdrive_utils.get_checkpoint_path():
            self._load_hyperdrive_checkpoint(sess)

        if self.config.mode == 'train':
            self.writer = tf.summary.FileWriter(self.config.log_dir, graph=tf.get_default_graph())

    def save(self, sess, global_step=None):
        saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        save_path = saver.save(sess, self.save_path, global_step=global_step)
        return save_path


    def _get_checkpoint_path(self, checkpoint_info):
        """ Return checkpoint path if specified else None 
            Handle cases where provided checkpoint path is dir or file."""
    
        checkpoint_path = None
        if checkpoint_info:
            if os.path.isdir(checkpoint_info):
                checkpoint_state = tf.train.get_checkpoint_state(checkpoint_info)
                checkpoint_path = checkpoint_state.model_checkpoint_path if checkpoint_state else None
            else:
                checkpoint_path = checkpoint_info
    
        return checkpoint_path

    def _load_hyperdrive_checkpoint(self, sess):
        """ Load hyperdrive provided model checkpoint. """

        checkpoint_info = hyperdrive_utils.get_checkpoint_path()
        restore_path = self._get_checkpoint_path(checkpoint_info)

        if restore_path:
            print("Loading HyperDrive model checkpoint from {}".format(restore_path))
            vars_ = {var.name.split(":")[0]: var for var in tf.global_variables()}
            saver = tf.train.Saver(vars_, max_to_keep=self.config.max_to_keep)
            saver.restore(sess, restore_path)
        else:
            print("Failed to load HyperDrive model checkpoint using {}".format(checkpoint_info))

    def _load(self, sess):
        config = self.config
        vars_ = {var.name.split(":")[0]: var for var in tf.global_variables()}
        if config.load_ema:
            ema = self.model.var_ema
            for var in tf.trainable_variables():
                del vars_[var.name.split(":")[0]]
                vars_[ema.average_name(var)] = var
        saver = tf.train.Saver(vars_, max_to_keep=config.max_to_keep)

        if config.load_path:
            save_path = config.load_path
        elif config.load_step > 0:
            save_path = os.path.join(config.save_dir, "{}-{}".format(config.model_name, config.load_step))
        else:
            save_dir = config.save_dir
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "cannot load checkpoint at {}".format(save_dir)
            save_path = checkpoint.model_checkpoint_path
        print("Loading saved model from {}".format(save_path))
        saver.restore(sess, save_path)

    def add_summary(self, summary, global_step):
        self.writer.add_summary(summary, global_step)

    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.add_summary(summary, global_step)

    def dump_eval(self, e, precision=2, path=None):
        assert isinstance(e, Evaluation)
        if self.config.dump_pickle:
            path = path or os.path.join(self.config.eval_dir, "{}-{}.pklz".format(e.data_type, str(e.global_step).zfill(6)))
            with gzip.open(path, 'wb', compresslevel=3) as fh:
                pickle.dump(e.dict, fh)
        else:
            path = path or os.path.join(self.config.eval_dir, "{}-{}.json".format(e.data_type, str(e.global_step).zfill(6)))
            with open(path, 'w') as fh:
                json.dump(short_floats(e.dict, precision), fh)

    def dump_answer(self, e, path=None):
        assert isinstance(e, Evaluation)
        path = path or os.path.join(self.config.answer_dir, "{}-{}.json".format(e.data_type, str(e.global_step).zfill(6)))
        with open(path, 'w') as fh:
            json.dump(e.id2answer_dict, fh)

