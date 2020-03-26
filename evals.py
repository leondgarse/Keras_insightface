import pickle
import os
import io
from tqdm import tqdm
from skimage.io import imread
from sklearn.preprocessing import normalize
import tensorflow as tf
import numpy as np
import glob2


class epoch_eval_callback(tf.keras.callbacks.Callback):
    def __init__(self, basic_model, test_bin_file, batch_size=128, save_model=None, eval_freq=1, flip=False):
        super(epoch_eval_callback, self).__init__()
        bins, issame_list = np.load(test_bin_file, encoding="bytes", allow_pickle=True)
        ds = tf.data.Dataset.from_tensor_slices(bins)
        _imread = lambda xx: tf.image.convert_image_dtype(tf.image.decode_jpeg(xx), dtype=tf.float32)
        ds = ds.map(_imread)
        self.ds = ds.batch(batch_size)
        self.test_issame = np.array(issame_list)
        self.test_names = os.path.splitext(os.path.basename(test_bin_file))[0]
        self.max_accuracy = 0
        self.steps = int(np.ceil(len(bins) / batch_size))
        self.save_model = save_model
        self.eval_freq = eval_freq
        self.flip = flip
        self.basic_model = basic_model

    # def on_batch_end(self, batch=0, logs=None):
    def on_epoch_end(self, epoch=0, logs=None):
        if epoch % self.eval_freq != 0:
            return
        dists = []
        embs = []
        tf.print("")
        for img_batch in tqdm(self.ds, "Evaluating " + self.test_names, total=self.steps):
            emb = self.basic_model.predict(img_batch)
            if self.flip:
                emb_f = self.basic_model.predict(tf.image.flip_left_right(img_batch))
                emb = (emb + emb_f) / 2
            embs.extend(emb)
        embs = np.array(embs)
        if np.isnan(embs).sum() != 0:
            tf.print("NAN in embs, not a good one")
            return
        embs = normalize(embs)
        embs_a = embs[::2]
        embs_b = embs[1::2]
        dists = (embs_a * embs_b).sum(1)

        tt = np.sort(dists[self.test_issame[: dists.shape[0]]])
        ff = np.sort(dists[np.logical_not(self.test_issame[: dists.shape[0]])])
        # self.tt = tt
        # self.ff = ff
        # self.embs = embs

        t_steps = int(0.1 * ff.shape[0])
        acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]])
        acc_max_indx = np.argmax(acc_count)
        acc_max = acc_count[acc_max_indx] / dists.shape[0]
        acc_thresh = ff[acc_max_indx - t_steps]

        tf.print(
            "\n>>>> %s evaluation max accuracy: %f, thresh: %f, previous max accuracy: %f"
            % (self.test_names, acc_max, acc_thresh, self.max_accuracy)
        )
        if acc_max > self.max_accuracy:
            tf.print(">>>> Improved = %f" % (acc_max - self.max_accuracy))
            self.max_accuracy = acc_max
            if self.save_model:
                save_name_base = "%s_basic_%s_epoch_" % (self.save_model, self.test_names)
                save_path_base = os.path.join("./checkpoints", save_name_base)
                for ii in glob2.glob(save_path_base + "*.h5"):
                    os.remove(ii)
                save_path = save_path_base + "%d_%f.h5" % (epoch, self.max_accuracy)
                tf.print("Saving model to: %s" % (save_path))
                self.basic_model.save(save_path)
