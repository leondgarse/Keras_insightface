#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from glob2 import glob
from skimage import transform
from skimage.io import imread, imsave
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Face_detection:
    def __init__(self):
        import insightface
        import cv2

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if len(cvd) > 0 and int(cvd) != -1:
            ctx = 0
        else:
            ctx = -1
        # self.det = insightface.model_zoo.face_detection.retinaface_r50_v1()
        self.det = insightface.model_zoo.face_detection.retinaface_mnet025_v1()
        self.det.prepare(ctx)
        self.cv2 = cv2

    def face_align_landmark(self, img, landmark, image_size=(112, 112), method="similar"):
        tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
        src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]],
            dtype=np.float32,
        )
        tform.estimate(landmark, src)
        M = tform.params[0:2, :]
        ndimage = self.cv2.warpAffine(img, M, image_size, borderValue=0.0)
        if len(ndimage.shape) == 2:
            ndimage = np.stack([ndimage, ndimage, ndimage], -1)
        else:
            ndimage = self.cv2.cvtColor(ndimage, self.cv2.COLOR_BGR2RGB)
        return ndimage

    def do_detect_in_image(self, image, image_format="RGB"):
        imm_BGR = self.cv2.imread(image)
        bboxes, pps = self.det.detect(imm_BGR)
        if len(bboxes) != 0:
            return self.face_align_landmark(imm_BGR, pps[0])
        else:
            return np.array([])


def detection_in_folder(data_path):
    while data_path.endswith("/"):
        data_path = data_path[:-1]
    imms = glob(os.path.join(data_path, "*/*"))
    dest_path = data_path + "_aligned_112_112"
    det = Face_detection()

    for imm in tqdm(imms, "Detecting"):
        nimage = det.do_detect_in_image(imm)
        if nimage.shape[0] != 0:
            file_name = os.path.basename(imm)
            class_name = os.path.basename(os.path.dirname(imm))
            save_dir = os.path.join(dest_path, class_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            imsave(os.path.join(save_dir, file_name), nimage)
        else:
            print(">>>> None face detected in image:", imm)
    print(">>>> Saved aligned face images in:", dest_path)
    return dest_path


def eval_folder(model_file, data_path, batch_size=128, save_embeddings=None, debug=True):
    if save_embeddings and os.path.exists(save_embeddings):
        print(">>>> Reloading from backup:", save_embeddings)
        aa = np.load(save_embeddings)
        embs, imm_classes, filenames = aa["embs"], aa["imm_classes"], aa["filenames"]
        embs, imm_classes = embs.astype("float32"), imm_classes.astype("int")

    else:
        img_shape = (112, 112)
        mm = tf.keras.models.load_model(model_file)
        img_gen = ImageDataGenerator().flow_from_directory(
            data_path, class_mode="binary", target_size=img_shape, batch_size=batch_size, shuffle=False
        )
        steps = int(np.ceil(img_gen.classes.shape[0] / img_gen.batch_size))
        filenames = np.array(img_gen.filenames)

        embs, imm_classes = [], []
        for _ in tqdm(range(steps), "Embedding"):
            imm, imm_class = img_gen.next()
            emb = mm((imm - 127.5) * 0.0078125)
            embs.extend(emb)
            imm_classes.extend(imm_class)
        embs, imm_classes = normalize(np.array(embs).astype("float32")), np.array(imm_classes).astype("int")
        if save_embeddings:
            print(">>>> Saving embeddings to:", save_embeddings)
            np.savez(save_embeddings, embs=embs, imm_classes=imm_classes, filenames=filenames)
    if save_embeddings:
        result_name = os.path.splitext(os.path.basename(save_embeddings))[0]
    else:
        result_name = os.path.splitext(os.path.basename(model_file))[0]

    register_ids = np.unique(imm_classes)
    if debug:
        print(">>>> [base info] embs:", embs.shape, "imm_classes:", imm_classes.shape, "register_ids:", register_ids.shape)
    try:
        import cupy as cp

        embs = cp.array(embs)
        dist_func = lambda aa, bb: cp.dot(aa, bb).get()
        if debug:
            print(">>>> Using cupy.")
    except:
        dist_func = lambda aa, bb: np.dot(aa, bb)

    pos_dists, neg_dists, register_base_dists = [], [], []
    for register_id in tqdm(register_ids, "Evaluating"):
        pick_cond = imm_classes == register_id
        pos_embs = embs[pick_cond]
        dists = dist_func(pos_embs, pos_embs.T)
        register_base = dists.sum(0).argmax()
        register_base_dist = dists[register_base]
        register_base_emb = embs[pick_cond][register_base]
        pos_dist = register_base_dist[np.arange(dists.shape[0]) != register_base]

        register_base_dist = dist_func(embs, register_base_emb)
        neg_dist = register_base_dist[imm_classes != register_id]

        pos_dists.extend(pos_dist)
        neg_dists.extend(neg_dist)
        register_base_dists.append(register_base_dist)
        # print(">>> [register info] register_id:", register_id, "register_base_img:", filenames[pick_cond][register_base])
    pos_dists = np.array(pos_dists).astype("float")
    neg_dists = np.array(neg_dists).astype("float")
    register_base_dists = np.array(register_base_dists).T
    if debug:
        print(
            ">>>> pos_dists:", pos_dists.shape, "neg_dists:", neg_dists.shape, "register_base_dists:", register_base_dists.shape
        )

    accuracy = (register_base_dists.argmax(1) == imm_classes).sum() / register_base_dists.shape[0]
    label = np.concatenate([np.ones_like(pos_dists), np.zeros_like(neg_dists)])
    score = np.concatenate([pos_dists, neg_dists])
    return accuracy, score, label


def plot_tpr_far(score, label):
    fpr, tpr, _ = roc_curve(label, score)
    roc_auc = auc(fpr, tpr)

    fpr_show = [10 ** (-ii) for ii in range(1, 7)[::-1]]
    fpr_reverse, tpr_reverse = fpr[::-1], tpr[::-1]
    tpr_show = [tpr_reverse[np.argmin(abs(fpr_reverse - ii))] for ii in fpr_show]
    print(pd.DataFrame({"FPR": fpr_show, "TPR": tpr_show}).set_index("FPR").T.to_markdown())

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(fpr, tpr, lw=1, label="[%s (AUC = %0.4f%%)]" % (result_name, roc_auc * 100))
        plt.xlim([10 ** -6, 0.1])
        plt.xscale("log")
        plt.xticks(fpr_show)
        plt.xlabel("False Positive Rate")
        plt.ylim([0, 1.0])
        plt.yticks(np.linspace(0, 1.0, 8, endpoint=True))
        plt.ylabel("True Positive Rate")

        plt.grid(linestyle="--", linewidth=1)
        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    except:
        print("matplotlib plot failed")
        fig = None
    return fig


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_path", type=str, default=None, help="Data path, containing images in class folders")
    parser.add_argument("-m", "--model_file", type=str, default=None, help="Model file, keras h5")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-D", "--detection", action="store_true", help="Run face detection before embedding")
    parser.add_argument("-S", "--save_embeddings", type=str, default=None, help="Save / Reload embeddings data")
    args = parser.parse_known_args(sys.argv[1:])[0]

    if args.model_file == None and args.data_path == None and args.save_embeddings == None:
        print(">>>> Please seee `--help` for usage")
        sys.exit(1)

    data_path = args.data_path
    if args.detection:
        data_path = detection_in_folder(args.data_path)
        print()
    accuracy, score, label = eval_folder(args.model_file, data_path, args.batch_size, args.save_embeddings, debug=True)
    print(">>>> top1 accuracy:", accuracy)
    plot_tpr_far(score, label)
elif __name__ == "__test__":
    data_path = "temp_test/faces_emore_test/"
    model_file = "checkpoints/TT_mobilenet_pointwise_distill_128_emb512_dr04_arc_bs400_r100_emore_fp16_basic_agedb_30_epoch_49_0.972333.h5"
    batch_size = 64
    save_embeddings = None
