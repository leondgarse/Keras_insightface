import sys
import select
import os
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.python.keras import backend as K


class Gently_stop_callback(keras.callbacks.Callback):
    def __init__(self, prompt="Continue? ([Y]/n)", time_out=3):
        super(Gently_stop_callback, self).__init__()
        self.yes_or_no = lambda: "n" not in self.timeout_input(prompt, time_out, default="y")[1].lower()

    def on_epoch_end(self, epoch, logs={}):
        print()
        if not self.yes_or_no():
            self.model.stop_training = True

    def timeout_input(self, prompt, timeout=3, default=""):
        print(prompt, end=": ", flush=True)
        inputs, outputs, errors = select.select([sys.stdin], [], [], timeout)
        print()
        return (0, sys.stdin.readline().strip()) if inputs else (-1, default)

class My_history(keras.callbacks.Callback):
    def __init__(self, initial_file=None, evals=[]):
        super(My_history, self).__init__()
        if initial_file and os.path.exists(initial_file):
            with open(initial_file, "r") as ff:
                self.history = json.load(ff)
        else:
            self.history = {}
        self.evals = evals
        self.initial_file = initial_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))
        for ee in self.evals:
            self.history.setdefault(ee.test_names, []).append(float(ee.cur_acc))
        if self.initial_file:
            with open(self.initial_file, "w") as ff:
                json.dump(self.history, ff)

    def print_hist(self):
        for kk, vv in self.history.items():
            print("  %s = %s" % (kk, vv))


class CosineLrScheduler(keras.callbacks.Callback):
    def __init__(self, lr_base, decay_steps, lr_min=0.0, warmup_iters=0, on_batch=False):
        super(CosineLrScheduler, self).__init__()
        self.lr_base, self.decay_steps, self.warmup_iters, self.on_batch = lr_base, decay_steps, warmup_iters, on_batch
        self.decay_steps -= self.warmup_iters
        self.schedule = keras.experimental.CosineDecay(self.lr_base, self.decay_steps, alpha=lr_min/lr_base)
        if on_batch == True:
            self.on_train_batch_begin = self.__lr_sheduler__
        else:
            self.on_epoch_begin = self.__lr_sheduler__

    def __lr_sheduler__(self, iterNum, logs=None):
        if iterNum < self.warmup_iters:
            lr = self.lr_base
        else:
            lr = self.schedule(iterNum - self.warmup_iters)
        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)
        if self.on_batch == False:
            print("\nLearning rate for epoch {} is {}".format(iterNum + 1, lr))
        return lr

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def scheduler(epoch, lr_base, decay_rate=0.05, lr_min=0):
    lr = lr_base if epoch < 10 else lr_base * np.exp(decay_rate * (10 - epoch))
    lr = lr_min if lr < lr_min else lr
    print("\nLearning rate for epoch {} is {}".format(epoch + 1, lr))
    return lr


def basic_callbacks(checkpoint="keras_checkpoints.h5", evals=[], lr=0.001, lr_decay=0.05, decay_type='exp', lr_min=0):
    checkpoint_base = "./checkpoints"
    if not os.path.exists(checkpoint_base):
        os.mkdir(checkpoint_base)
    checkpoint = os.path.join(checkpoint_base, checkpoint)
    model_checkpoint = ModelCheckpoint(checkpoint, verbose=1)

    if decay_type.lower().startswith("exp"):
        ss = lambda epoch: scheduler(epoch, lr, lr_decay, lr_min)
        lr_scheduler = LearningRateScheduler(ss)
    else:
        lr_scheduler = CosineLrScheduler(lr_base=lr, decay_steps=lr_decay, lr_min=lr_min, warmup_iters=10, on_batch=False)
    my_history = My_history(os.path.splitext(checkpoint)[0] + "_hist.json", evals=evals)
    return [model_checkpoint, lr_scheduler, my_history, Gently_stop_callback()]
