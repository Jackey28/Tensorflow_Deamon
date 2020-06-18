import json
import sys
import os
from datetime import datetime
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class LogHook(tf.train.SessionRunHook):
    def __init__(self, log_path, log_file, log_every_n_iter, tensor_dict, stat_tensor_dict=None, total_size=0,
                 hash_size=0):
        if stat_tensor_dict is None:
            stat_tensor_dict = {}
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_file = open('{}/{}'.format(log_path, log_file), 'w')
        self.log_every_n_iter = log_every_n_iter
        assert len(tensor_dict) != 0
        assert isinstance(tensor_dict, dict)
        self.tensor_dict = {}
        self.tensor_dict.update(tensor_dict)
        self.format_str = '%s: %s'
        self.tensor_dict.update({'steps': tf.train.get_global_step()})
        self.value_dict = {}


    def end(self, session):
        self.log_data()
        for key in self.change_value_cond.keys():
            self.log_file.write('%s_record:%s\n' % (key, self.stat_record_dict[key]))
        self.log_file.close()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.tensor_dict)

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        res = run_values.results
        for key in self.tensor_dict:
            self.value_dict[key] = res[key]
        for key in self.stat_value_dict:
            if self.change_value_cond[key](res[key], self.stat_value_dict[key]):
                self.stat_value_dict[key] = res[key]
                self.stat_record_dict[key] = self.log_data(False)
        if 'avg' in res and res['avg'] == self.total_size / self.hash_size:
            self.log_data()
            #run_context.request_stop()
        if res['steps'] % self.log_every_n_iter == 0:
            self.log_data()


    def log_data(self, show=True):
        cur_time = datetime.now()
        values_string = str(self.value_dict)
        log_message = self.format_str % (cur_time, values_string)

        if show:
            print(log_message)
            self.log_file.write(log_message + '\n')
        return log_message




class train_hooks(tf.train.SessionRunHook):
    def __init__(self, tensor_dict):
        self.tensor_dict = tensor_dict

    def begin(self):
        return
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.tensor_dict)
    def after_create_session(self, session, coord):
        return
    def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
        print(run_values)
    def end(self, session):
        print("end")
    def log_data(self):
        real_time = datetime.now()
        print(real_time)
