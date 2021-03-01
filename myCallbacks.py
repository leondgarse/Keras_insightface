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
        logs.pop("lr", None)
        self.history.setdefault("lr", []).append(float(self.model.optimizer.lr))
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
            self.history.setdefault("regular_loss", []).append(float(regular_loss))
            self.history["loss"][-1] -= regular_loss

        if self.initial_file:
            with open(self.initial_file, "w") as ff:
                json.dump(self.history, ff)

    def print_hist(self):
        print("{")
        for kk, vv in self.history.items():
            print("  '%s': %s," % (kk, vv))
        print("}")


class OptimizerWeightDecay(keras.callbacks.Callback):
    def __init__(self, lr_base, wd_base, is_lr_on_batch=False):
        super(OptimizerWeightDecay, self).__init__()
        self.wd_m = wd_base / lr_base
        self.lr_base, self.wd_base = lr_base, wd_base
        # self.model.optimizer.weight_decay = lambda: wd_m * self.model.optimizer.lr
        self.is_lr_on_batch = is_lr_on_batch
        if is_lr_on_batch:
            self.on_train_batch_begin = self.__update_wd__
        else:
            self.on_epoch_begin = self.__update_wd__

    def __update_wd__(self, step, log=None):
        if self.model is not None:
            wd = self.wd_m * K.get_value(self.model.optimizer.lr)
            # wd = self.wd_base * K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.weight_decay, wd)
        # wd = self.model.optimizer.weight_decay
        if not self.is_lr_on_batch or step == 0:
            print("Weight decay is {}".format(wd))


class ConstantDecayScheduler(keras.callbacks.Callback):
    def __init__(self, lr_base, lr_decay_steps, decay_rate=0.1):
        super(ConstantDecayScheduler, self).__init__()
        self.lr_decay_steps, self.lr_base, self.decay_rate = lr_decay_steps, lr_base, decay_rate

    def on_epoch_begin(self, step, log=None):
        lr = self.constant_decay(step)
        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)
        print("\nLearning rate for iter {} is {}".format(step + 1, lr))
        return lr

    def constant_decay(self, cur_step):
        for id, ii in enumerate(self.lr_decay_steps):
            if cur_step < ii:
                return self.lr_base * self.decay_rate ** id
        return self.lr_base * self.decay_rate ** len(self.lr_decay_steps)


class CosineLrScheduler(keras.callbacks.Callback):
    def __init__(
        self,
        lr_base,
        first_restart_step,
        m_mul=0.5,
        t_mul=1.0,
        lr_min=1e-5,
        warmup=0,
        decay_step=100,
        steps_per_epoch=-1,
        keep_as_min=1,
    ):
        super(CosineLrScheduler, self).__init__()
        self.lr_base, self.m_mul, self.t_mul, self.lr_min = lr_base, m_mul, t_mul, lr_min
        self.first_restart_step = first_restart_step
        self.warmup, self.decay_step, self.keep_as_min = warmup, decay_step, keep_as_min
        self.steps_per_epoch = steps_per_epoch  # Set after dataset inited
        self.init_step_num = 0
        self.cur_epoch = 0
        self.is_built = False

    def build(self):
        if self.first_restart_step < 500:
            # first_restart_step is epoch number, will nultiply with steps_per_epoch
            self.first_restart_step *= self.steps_per_epoch

        if self.warmup < 500:
            self.warmup *= self.steps_per_epoch

        if self.keep_as_min < 500:
            self.keep_as_min *= self.steps_per_epoch

        self.start_keep_as_min, self.stop_keep_as_min = [], []
        self.keep_as_min_batchs_already, self.is_keeping_as_min = 0, False
        alpha = self.lr_min / self.lr_base
        first_restart_step = self.first_restart_step // self.decay_step
        if self.lr_min == self.lr_base * self.m_mul:
            self.schedule = keras.experimental.CosineDecay(self.lr_base, first_restart_step, alpha=alpha)
        else:
            # with `first_restart_step, t_mul, warmup = 10, 2, 1` restart epochs will be:
            # ee = lambda ss: warmup + first_restart_step * np.sum([t_mul ** jj for jj in range(ss)])
            # [ee(ii) for ii in range(1, 5)] == [11, 31, 71, 151]
            self.schedule = keras.experimental.CosineDecayRestarts(
                self.lr_base, first_restart_step, t_mul=self.t_mul, m_mul=self.m_mul, alpha=alpha
            )
            if self.keep_as_min != 0 and self.lr_min != 0:
                restart_mul = [np.sum([self.t_mul ** jj for jj in range(ii)]) for ii in range(1, 5)]
                restart_batch_nums = [self.warmup + self.first_restart_step * ii for ii in restart_mul]
                self.start_keep_as_min = [int(ii + self.keep_as_min * id) for id, ii in enumerate(restart_batch_nums)]
                self.stop_keep_as_min = [int(ii + self.keep_as_min) for ii in self.start_keep_as_min]

        if self.warmup != 0:
            # self.warmup_lr_func = lambda ii: self.lr_base
            self.warmup_lr_func = lambda ii: self.lr_min + (self.lr_base - self.lr_min) * ii / self.warmup

    def on_epoch_begin(self, cur_epoch, logs=None):
        if not self.is_built:
            self.build()
            self.is_built = True
        self.init_step_num = int(self.steps_per_epoch * cur_epoch)
        self.cur_epoch = cur_epoch

    def on_train_batch_begin(self, iterNum, logs=None):
        global_iterNum = iterNum + self.init_step_num
        if not self.is_keeping_as_min and global_iterNum in self.start_keep_as_min:
            print(">>>> Keep lr as min:", self.lr_min, ", global_iterNum:", global_iterNum)
            self.keep_as_min_batchs_already += self.keep_as_min
            self.is_keeping_as_min = True
        elif self.is_keeping_as_min and global_iterNum in self.stop_keep_as_min:
            self.is_keeping_as_min = False

        if self.is_keeping_as_min:
            lr = self.lr_min
        elif global_iterNum < self.warmup:
            lr = self.warmup_lr_func(global_iterNum)
        else:
            lr = self.schedule((global_iterNum - self.warmup - self.keep_as_min_batchs_already) // self.decay_step)
        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)
        if iterNum == 0:
            print("\nLearning rate for iter {} is {}".format(self.cur_epoch + 1, lr))
        return lr


def scheduler_warmup(lr_target, cur_epoch, lr_init=0.1, epochs=10):
    stags = int(np.log10(lr_init / lr_target)) + 1
    steps = epochs / stags
    lr = lr_init
    while cur_epoch > steps:
        lr *= 0.1
        cur_epoch -= steps
    return lr


def exp_scheduler(epoch, lr_base, decay_rate=0.05, lr_min=0, warmup=10):
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
        lr_scheduler = ConstantDecayScheduler(lr_base=lr, lr_decay_steps=lr_decay_steps, decay_rate=lr_decay)
    elif lr_decay_steps > 1:
        # Cosine decay on epoch / batch
        lr_scheduler = CosineLrScheduler(lr_base=lr, first_restart_step=lr_decay_steps, m_mul=lr_decay, lr_min=lr_min)
    else:
        # Exponential decay
        warmup = 10
        lr_scheduler = LearningRateScheduler(lambda epoch: exp_scheduler(epoch, lr, lr_decay, lr_min, warmup=warmup))
    my_history = My_history(os.path.splitext(checkpoint)[0] + "_hist.json", evals=evals)
    # tensor_board_log = keras.callbacks.TensorBoard(log_dir=os.path.splitext(checkpoint)[0] + '_logs')
    return [my_history, model_checkpoint, lr_scheduler, Gently_stop_callback()]
