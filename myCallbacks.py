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
            self.history.setdefault(ee.test_names + "_thresh", []).append(float(ee.acc_thresh))
        for kk, vv in self.custom_obj.items():
            tt = losses_utils.compute_weighted_loss(vv())
            self.history.setdefault(kk, []).append(tt)
        if len(self.model.losses) != 0:
            regular_loss = K.sum(self.model.losses).numpy()
            print("regular_loss:", regular_loss)
            self.history.setdefault("regular_loss", []).append(float(regular_loss))
            self.history["loss"][-1] -= regular_loss
        if self.initial_file:
            with open(self.initial_file, "w") as ff:
                json.dump(self.history, ff)

    def print_hist(self):
        for kk, vv in self.history.items():
            print("  %s = %s" % (kk, vv))


class OptimizerWeightDecay(keras.callbacks.Callback):
    def __init__(self, lr_base, wd_base):
        super(OptimizerWeightDecay, self).__init__()
        self.wd_m = wd_base / lr_base
        self.lr_base, self.wd_base = lr_base, wd_base
        # self.model.optimizer.weight_decay = lambda: wd_m * self.model.optimizer.lr

    def on_epoch_begin(self, step, log=None):
        if self.model is not None:
            wd = self.wd_m * K.get_value(self.model.optimizer.lr)
            # wd = self.wd_base * K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.weight_decay, wd)
        # wd = self.model.optimizer.weight_decay
        print("Weight decay for iter {} is {}".format(step + 1, wd))


class ConstantDecayScheduler(keras.callbacks.Callback):
    def __init__(self, lr_decay_steps, lr_base=1e-1, decay_rate=0.1):
        super(ConstantDecayScheduler, self).__init__()
        self.lr_decay_steps, self.lr_base, self.decay_rate = lr_decay_steps, lr_base, decay_rate

    def on_epoch_begin(self, step, log=None):
        lr = self.constant_decay(step)
        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)
        print("\nLearning rate for iter {} is {}".format(step + 1, lr))
        return lr

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.lr)

    def constant_decay(self, cur_step):
        for id, ii in enumerate(self.lr_decay_steps):
            if cur_step < ii:
                return self.lr_base * self.decay_rate ** id
        return self.lr_base * self.decay_rate ** len(self.lr_decay_steps)


class CosineLrScheduler(keras.callbacks.Callback):
    def __init__(self, lr_base, decay_steps, lr_min=0.0, warmup_iters=0, lr_on_batch=0, restarts=1, m_mul=0.5):
        super(CosineLrScheduler, self).__init__()
        self.lr_base, self.decay_steps, self.lr_min = lr_base, decay_steps, lr_min
        if restarts > 1:
            # with restarts == 3, t_mul == 2, restart_step 1 + 2 + 4 == 7
            restart_step = sum([2 ** ii for ii in range(restarts)]) * max(1, lr_on_batch)
            self.schedule = keras.experimental.CosineDecayRestarts(
                lr_base, self.decay_steps // restart_step, t_mul=2.0, m_mul=m_mul, alpha=lr_min / lr_base
            )
        else:
            self.schedule = keras.experimental.CosineDecay(lr_base, self.decay_steps, alpha=lr_min / lr_base)
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
        logs["lr"] = K.get_value(self.model.optimizer.lr)


def scheduler_warmup(lr_target, cur_epoch, lr_init=0.1, epochs=10):
    stags = int(np.log10(lr_init / lr_target)) + 1
    steps = epochs / stags
    lr = lr_init
    while cur_epoch > steps:
        lr *= 0.1
        cur_epoch -= steps
    return lr


def scheduler(epoch, lr_base, decay_rate=0.05, lr_min=0, warmup=10):
    lr = lr_base if epoch < warmup else lr_base * np.exp(decay_rate * (warmup - epoch))
    # lr = scheduler_warmup(lr_base, epoch) if epoch < warmup else lr_base * np.exp(decay_rate * (warmup - epoch))
    lr = lr_min if lr < lr_min else lr
    print("\nLearning rate for iter {} is {}".format(epoch + 1, lr))
    return lr


def basic_callbacks(checkpoint="keras_checkpoints.h5", evals=[], lr=0.001, lr_decay=0.05, lr_min=0, lr_decay_steps=0):
    checkpoint_base = "./checkpoints"
    if not os.path.exists(checkpoint_base):
        os.mkdir(checkpoint_base)
    checkpoint = os.path.join(checkpoint_base, checkpoint)
    model_checkpoint = ModelCheckpoint(checkpoint, verbose=1)
    # model_checkpoint = keras.callbacks.experimental.BackupAndRestore(checkpoint_base)

    if isinstance(lr_decay_steps, list):
        # Constant decay on epoch
        lr_scheduler = ConstantDecayScheduler(lr_decay_steps=lr_decay_steps, lr_base=lr, decay_rate=lr_decay)
    elif lr_decay < 1:
        # Exponential decay
        warmup = 10 if lr_decay_steps == 0 else lr_decay_steps
        lr_scheduler = LearningRateScheduler(lambda epoch: scheduler(epoch, lr, lr_decay, lr_min, warmup=warmup))
    else:
        # Cosine decay on epoch / batch
        lr_scheduler = CosineLrScheduler(
            lr_base=lr, decay_steps=lr_decay, lr_min=lr_min, warmup_iters=1, lr_on_batch=lr_decay_steps, restarts=4
        )
    my_history = My_history(os.path.splitext(checkpoint)[0] + "_hist.json", evals=evals)
    # tensor_board_log = keras.callbacks.TensorBoard(log_dir=os.path.splitext(checkpoint)[0] + '_logs')
    return [my_history, model_checkpoint, lr_scheduler, Gently_stop_callback()]
