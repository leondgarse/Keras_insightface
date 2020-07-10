import sys
import select
import os
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils

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
        self.custom_obj = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            k = "accuracy" if "accuracy" in k else k
            self.history.setdefault(k, []).append(float(v))
        for ee in self.evals:
            self.history.setdefault(ee.test_names, []).append(float(ee.cur_acc))
        for kk, vv in self.custom_obj.items():
            tt = losses_utils.compute_weighted_loss(vv())
            self.history.setdefault(kk, []).append(tt)
        if self.initial_file:
            with open(self.initial_file, "w") as ff:
                json.dump(self.history, ff)

    def print_hist(self):
        for kk, vv in self.history.items():
            print("  %s = %s" % (kk, vv))


class CosineLrScheduler(keras.callbacks.Callback):
    def __init__(self, lr_base, decay_steps, lr_min=0.0, warmup_iters=0, lr_on_batch=0, restarts=1):
        super(CosineLrScheduler, self).__init__()
        self.lr_base, self.decay_steps, self.lr_min= lr_base, decay_steps, lr_min
        if restarts > 1:
            # with restarts == 3, t_mul == 2, restart_step 1 + 2 + 4 == 7
            restart_step = sum([2 ** ii for ii in range(restarts)]) * max(1, lr_on_batch)
            self.schedule = keras.experimental.CosineDecayRestarts(lr_base, self.decay_steps // restart_step, t_mul=2.0, m_mul=0.5, alpha=lr_min / lr_base)
        else:
            self.schedule = keras.experimental.CosineDecay(lr_base, self.decay_steps, alpha=lr_min/lr_base)
        if lr_on_batch < 1:
            self.on_epoch_begin = self.__lr_sheduler__
        else:
            self.on_train_batch_begin = self.__lr_sheduler__
        if warmup_iters != 0:
            # self.warmup_lr_func = lambda ii: lr_min + (lr_base - lr_min) * ii / warmup_iters
            self.warmup_lr_func = lambda ii: lr_base
        self.lr_on_batch = max(1, lr_on_batch)
        self.warmup_iters = warmup_iters / self.lr_on_batch

    def __lr_sheduler__(self, iterNum, logs=None):
        iterNum //= self.lr_on_batch
        if iterNum < self.warmup_iters:
            lr = self.warmup_lr_func(iterNum)
        else:
            lr = self.schedule(iterNum - self.warmup_iters)
        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)
        print("\nLearning rate for iter {} is {}".format(iterNum + 1, lr))
        return lr

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def scheduler(epoch, lr_base, decay_rate=0.05, lr_min=0):
    lr = lr_base if epoch < 10 else lr_base * np.exp(decay_rate * (10 - epoch))
    lr = lr_min if lr < lr_min else lr
    print("\nLearning rate for epoch {} is {}".format(epoch + 1, lr))
    return lr


def basic_callbacks(checkpoint="keras_checkpoints.h5", evals=[], lr=0.001, lr_decay=0.05, lr_min=0, lr_on_batch=0):
    checkpoint_base = "./checkpoints"
    if not os.path.exists(checkpoint_base):
        os.mkdir(checkpoint_base)
    checkpoint = os.path.join(checkpoint_base, checkpoint)
    model_checkpoint = ModelCheckpoint(checkpoint, verbose=1)

    if lr_decay < 1:
        # Exponential decay
        lr_scheduler = LearningRateScheduler(lambda epoch: scheduler(epoch, lr, lr_decay, lr_min))
    else:
        # Cosine decay on epoch / batch
        lr_scheduler = CosineLrScheduler(lr_base=lr, decay_steps=lr_decay, lr_min=lr_min, warmup_iters=0, lr_on_batch=lr_on_batch, restarts=3)
    my_history = My_history(os.path.splitext(checkpoint)[0] + "_hist.json", evals=evals)
    # tensor_board_log = keras.callbacks.TensorBoard(log_dir=os.path.splitext(checkpoint)[0] + '_logs')
    return [model_checkpoint, lr_scheduler, my_history, Gently_stop_callback()]
