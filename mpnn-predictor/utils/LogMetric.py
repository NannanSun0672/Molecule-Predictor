#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
from tensorboard_logger import configure, log_value


def error_ratio(pred, target):
    if type(pred) is not np.ndarray:
        pred = np.array(pred)
    if type(target) is not np.ndarray:
        target = np.array(target)

    return np.mean(np.divide(np.abs(pred - target), np.abs(target)))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.isdir(self.log_dir):
            # if the directory does not exist we create the directory
            os.makedirs(self.log_dir)
        else:
            # clean previous logged data under the same directory name
            self._remove(self.log_dir)

        # configure the project
        configure(self.log_dir)

        self.global_step = 0

    def log_value(self, name, value):
        #log_value(name, value, self.global_step)
        #return self
        path = os.path.join(self.log_dir,name)+".log"
        with open(path,"a")as fw:
            fw.write(name+"\t"+str(value)+"\n")
    def log_lr(self,name,value):
        log_value(name,value,self.global_step)
        return self
    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)  # remove dir and all contains
