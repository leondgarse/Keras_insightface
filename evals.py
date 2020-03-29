import pickle
import os
import io
from tqdm import tqdm
from skimage.io import imread
from sklearn.preprocessing import normalize
import tensorflow as tf
import numpy as np
import glob2
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
from sklearn.decomposition import PCA


class epoch_eval_callback(tf.keras.callbacks.Callback):
    def __init__(self, basic_model, test_bin_file, batch_size=128, save_model=None, eval_freq=1, flip=False, PCA_acc=True):
        super(epoch_eval_callback, self).__init__()
        bins, issame_list = np.load(test_bin_file, encoding="bytes", allow_pickle=True)
        ds = tf.data.Dataset.from_tensor_slices(bins)
        _imread = lambda xx: tf.image.convert_image_dtype(tf.image.decode_jpeg(xx), dtype=tf.float32)
        ds = ds.map(_imread)
        self.ds = ds.batch(batch_size)
        self.test_issame = np.array(issame_list)
        self.test_names = os.path.splitext(os.path.basename(test_bin_file))[0]
        self.steps = int(np.ceil(len(bins) / batch_size))
        self.basic_model = basic_model
        self.max_accuracy, self.cur_acc = 0.0, 0.0
        self.save_model, self.eval_freq, self.flip, self.PCA_acc = save_model, eval_freq, flip, PCA_acc

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
        _, _, accuracy, val, val_std, far = evaluate(embs, self.test_issame, nrof_folds=10)
        acc2, std2 = np.mean(accuracy), np.std(accuracy)
        embs_a = embs[::2]
        embs_b = embs[1::2]
        dists = (embs_a * embs_b).sum(1)

        tt = np.sort(dists[self.test_issame[: dists.shape[0]]])
        ff = np.sort(dists[np.logical_not(self.test_issame[: dists.shape[0]])])
        self.tt = tt
        self.ff = ff
        self.embs = embs

        t_steps = int(0.1 * ff.shape[0])
        acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]])
        acc_max_indx = np.argmax(acc_count)
        acc_max = acc_count[acc_max_indx] / dists.shape[0]
        acc_thresh = ff[acc_max_indx - t_steps]
        self.cur_acc = acc_max

        tf.print(
            "\n>>>> %s evaluation max accuracy: %f, thresh: %f, previous max accuracy: %f, PCA accuray = %f ± %f"
            % (self.test_names, acc_max, acc_thresh, self.max_accuracy, acc2, std2)
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


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print("doing pca on", fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set]
            )
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind="slinear")
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca
    )
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds
    )
    return tpr, fpr, accuracy, val, val_std, far
