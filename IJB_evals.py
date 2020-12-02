import os
import numpy as np
from tqdm import tqdm
from skimage import transform
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
import pandas as pd
import cv2


class Mxnet_model_interf:
    def __init__(self, model_file, layer="fc1", image_size=(112, 112)):
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if len(cvd) > 0 and int(cvd) != -1:
            ctx = [mx.gpu(ii) for ii in range(len(cvd.split(",")))]
        else:
            ctx = [mx.cpu()]

        prefix, epoch = model_file.split(",")
        print(">>>> loading mxnet model:", prefix, epoch, ctx)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        sym = all_layers[layer + "_output"]
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[("data", (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2)
        data = mx.nd.array(imgs)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()
        return emb


def keras_model_interf(model_file):
    mm = tf.keras.models.load_model(model_file, compile=False)
    return lambda imgs: mm((tf.cast(imgs, "float32") - 127.5) * 0.0078125).numpy()


def face_align_landmark(img, landmark, image_size=(112, 112), method="similar"):
    tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
    src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32
    )
    tform.estimate(landmark, src)
    # ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
    # ndimage = (ndimage * 255).astype(np.uint8)
    M = tform.params[0:2, :]
    ndimage = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    if len(ndimage.shape) == 2:
        ndimage = np.stack([ndimage, ndimage, ndimage], -1)
    else:
        ndimage = cv2.cvtColor(ndimage, cv2.COLOR_BGR2RGB)
    return ndimage


def extract_IJB_data(data_path, subset, save_path=None, force_reload=False):
    if save_path == None:
        save_path = os.path.join(data_path, subset + "_backup.npz")
    if not force_reload and os.path.exists(save_path):
        print(">>>> Reloading from backup: %s..." % save_path)
        aa = np.load(save_path)
        return (
            aa["templates"],
            aa["medias"],
            aa["p1"],
            aa["p2"],
            aa["label"],
            aa["img_names"],
            aa["landmarks"],
            aa["face_scores"],
        )

    if subset == "IJBB":
        media_list_path = os.path.join(data_path, "IJBB/meta/ijbb_face_tid_mid.txt")
        pair_list_path = os.path.join(data_path, "IJBB/meta/ijbb_template_pair_label.txt")
        img_path = os.path.join(data_path, "IJBB/loose_crop")
        img_list_path = os.path.join(data_path, "IJBB/meta/ijbb_name_5pts_score.txt")
    else:
        media_list_path = os.path.join(data_path, "IJBC/meta/ijbc_face_tid_mid.txt")
        pair_list_path = os.path.join(data_path, "IJBC/meta/ijbc_template_pair_label.txt")
        img_path = os.path.join(data_path, "IJBC/loose_crop")
        img_list_path = os.path.join(data_path, "IJBC/meta/ijbc_name_5pts_score.txt")

    print(">>>> Loading templates and medias...")
    ijb_meta = np.loadtxt(media_list_path, dtype=str)  # ['1.jpg', '1', '69544']
    templates, medias = ijb_meta[:, 1].astype(np.int), ijb_meta[:, 2].astype(np.int)
    print(
        ">>>> Loaded templates: %s, medias: %s, unique templates: %s"
        % (templates.shape, medias.shape, np.unique(templates).shape)
    )
    # (227630,) (227630,) (12115,)

    print(">>>> Loading pairs...")
    pairs = np.loadtxt(pair_list_path, dtype=str)  # ['1', '11065', '1']
    p1, p2, label = pairs[:, 0].astype(np.int), pairs[:, 1].astype(np.int), pairs[:, 2].astype(np.int)
    print(">>>> Loaded p1: %s, unique p1: %s" % (p1.shape, np.unique(p1).shape))
    print(">>>> Loaded p2: %s, unique p2: %s" % (p2.shape, np.unique(p2).shape))
    print(">>>> Loaded label: %s, label value counts: %s" % (label.shape, dict(zip(*np.unique(label, return_counts=True)))))
    # (8010270,) (8010270,) (8010270,) (1845,) (10270,) # 10270 + 1845 = 12115
    # {0: 8000000, 1: 10270}

    print(">>>> Loading images...")
    with open(img_list_path, "r") as ff:
        # 1.jpg 46.060 62.026 87.785 60.323 68.851 77.656 52.162 99.875 86.450 98.648 0.999
        img_records = np.array([ii.strip().split(" ") for ii in ff.readlines()])

    img_names = np.array([os.path.join(img_path, ii) for ii in img_records[:, 0]])
    landmarks = img_records[:, 1:-1].astype("float32").reshape(-1, 5, 2)
    face_scores = img_records[:, -1].astype("float32")
    print(">>>> Loaded img_names: %s, landmarks: %s, face_scores: %s" % (img_names.shape, landmarks.shape, face_scores.shape))
    # (227630,) (227630, 5, 2) (227630,)
    print(">>>> Loaded face_scores value counts:", dict(zip(*np.histogram(face_scores, bins=9)[::-1])))
    # {0.1: 2515, 0.2: 0, 0.3: 62, 0.4: 94, 0.5: 136, 0.6: 197, 0.7: 291, 0.8: 538, 0.9: 223797}

    # print(">>>> Running warp affine...")
    # ndimages = [
    #     face_align_landmark(cv2.imread(img_name), landmark)
    #     for img_name, landmark in tqdm(zip(img_names, landmarks), total=len(img_names))
    # ]
    # ndimages = np.stack(ndimages)
    # print("Finale image size:", ndimages.shape)
    # (227630, 112, 112, 3)

    print(">>>> Saving backup to: %s..." % save_path)
    np.savez(
        save_path,
        templates=templates,
        medias=medias,
        p1=p1,
        p2=p2,
        label=label,
        img_names=img_names,
        landmarks=landmarks,
        face_scores=face_scores,
    )
    return templates, medias, p1, p2, label, img_names, landmarks, face_scores


def get_embeddings(model_interf, img_names, landmarks, batch_size=64, flip=True):
    steps = int(np.ceil(len(img_names) / batch_size))
    embs, embs_f = [], []
    for batch_id in tqdm(range(0, len(img_names), batch_size), "Embedding", total=steps):
        batch_imgs, batch_landmarks = img_names[batch_id : batch_id + batch_size], landmarks[batch_id : batch_id + batch_size]
        ndimages = [face_align_landmark(cv2.imread(img), landmark) for img, landmark in zip(batch_imgs, batch_landmarks)]
        ndimages = np.stack(ndimages)
        embs.extend(model_interf(ndimages))
        if flip:
            embs_f.extend(model_interf(ndimages[:, :, ::-1, :]))
    return np.array(embs), np.array(embs_f)


def process_embeddings(embs, embs_f=[], use_flip_test=True, use_norm_score=False, use_detector_score=True, face_scores=None):
    if use_flip_test and len(embs_f) != 0:
        embs = embs + embs_f
    if use_norm_score:
        embs = normalize(embs)
    if use_detector_score and face_scores is not None:
        embs = embs * np.expand_dims(face_scores, -1)
    return embs


def image2template_feature(img_feats=None, templates=None, medias=None):
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in tqdm(enumerate(unique_templates), "Extract template feature", total=len(unique_templates)):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for uu, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == uu)
            if ct == 1:
                media_norm_feats.append(face_norm_feats[ind_m])
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats.append(np.mean(face_norm_feats[ind_m], 0, keepdims=True))
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
    template_norm_feats = normalize(template_feats)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None, batch_size=100000):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    steps = int(np.ceil(len(p1) / batch_size))
    score = []
    for id in tqdm(range(steps), "Verification"):
        feat1 = template_norm_feats[template2id[p1[id * batch_size : (id + 1) * batch_size]].flatten()]
        feat2 = template_norm_feats[template2id[p2[id * batch_size : (id + 1) * batch_size]].flatten()]
        score.extend(np.sum(feat1 * feat2, -1))
    return np.array(score)


def run_model_test(
    data_path,
    subset,
    interf_func,
    batch_size=64,
    force_reload=False,
    use_flip_test=True,
    use_norm_score=False,
    use_detector_score=True,
):
    templates, medias, p1, p2, label, img_names, landmarks, face_scores = extract_IJB_data(
        data_path, subset, force_reload=force_reload
    )
    embs, embs_f = get_embeddings(interf_func, img_names, landmarks, batch_size=batch_size)
    img_input_feats = process_embeddings(
        embs,
        embs_f,
        use_flip_test=use_flip_test,
        use_norm_score=use_norm_score,
        use_detector_score=use_detector_score,
        face_scores=face_scores,
    )

    template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
    score = verification(template_norm_feats, unique_templates, p1, p2)
    return score, embs, embs_f, templates, medias, p1, p2, label, face_scores


def run_model_test_bunch(data_path, subset, interf_func, batch_size=64, force_reload=False):
    templates, medias, p1, p2, label, img_names, landmarks, face_scores = extract_IJB_data(
        data_path, subset, force_reload=force_reload
    )
    embs, embs_f = get_embeddings(interf_func, img_names, landmarks, batch_size=batch_size)

    results = {}
    for use_norm_score in [True, False]:
        for use_detector_score in [True, False]:
            for use_flip_test in [True, False]:
                name = "N{:d}D{:d}F{:d}".format(use_norm_score, use_detector_score, use_flip_test)
                print(">>>>", name, use_norm_score, use_detector_score, use_flip_test)

                img_input_feats = process_embeddings(
                    embs,
                    embs_f,
                    use_flip_test=use_flip_test,
                    use_norm_score=use_norm_score,
                    use_detector_score=use_detector_score,
                    face_scores=face_scores,
                )
                template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
                score = verification(template_norm_feats, unique_templates, p1, p2)
                results[name] = score
    return results, embs, embs_f, templates, medias, p1, p2, label, face_scores


def plot_roc_and_calculate_tpr(scores, names=None, label=None):
    score_dict = {}
    for id, score in enumerate(scores):
        name = None if names is None else names[id]
        if isinstance(score, str) and score.endswith(".npz"):
            aa = np.load(score)
            score = aa["scores"] if "scores" in aa else []
            label = aa["label"] if label is None and "label" in aa else label
            score_name = aa["names"] if "names" in aa else []
            for ss, nn in zip(score, score_name):
                score_dict[nn] = ss
        elif isinstance(score, str) and score.endswith(".npy"):
            name = name if name is not None else os.path.splitext(os.path.basename(score))[0]
            score_dict[name] = np.load(score)
        else:
            name = name if name is not None else str(id)
            score_dict[name] = score
    if label is None:
        print("Error: Label data is not provided")
        return None, None

    x_labels = [10 ** (-ii) for ii in range(1, 7)[::-1]]
    fpr_dict, tpr_dict, roc_auc_dict, tpr_result = {}, {}, {}, {}
    for name, score in score_dict.items():
        fpr, tpr, _ = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)
        fpr, tpr = np.flipud(fpr), np.flipud(tpr)  # select largest tpr at same fpr
        tpr_result[name] = [tpr[np.argmin(abs(fpr - ii))] for ii in x_labels]
        fpr_dict[name], tpr_dict[name], roc_auc_dict[name] = fpr, tpr, roc_auc
    tpr_result_df = pd.DataFrame(tpr_result, index=x_labels).T
    tpr_result_df.columns.name = "Methods"
    print(tpr_result_df.to_markdown())
    # print(tpr_result_df)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for name in score_dict:
            plt.plot(fpr_dict[name], tpr_dict[name], lw=1, label="[%s (AUC = %0.4f%%)]" % (name, roc_auc_dict[name] * 100))

        plt.xlim([10 ** -6, 0.1])
        plt.ylim([0.3, 1.0])
        plt.grid(linestyle="--", linewidth=1)
        plt.xticks(x_labels)
        plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
        plt.xscale("log")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC on IJB")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    except:
        print("Missing matplotlib")
        fig = None

    return tpr_result_df, fig


def parse_arguments(argv):
    import argparse
    default_save_result_name = "{model_name}_{subset}.npz"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_file", type=str, default=None, help="Saved model file path, could be keras / mxnet one")
    parser.add_argument("-d", "--data_path", type=str, default="./", help="Dataset path")
    parser.add_argument("-s", "--subset", type=str, default="IJBB", help="Subset test target, could be IJBB / IJBC")
    parser.add_argument("-r", "--save_result", type=str, default=default_save_result_name, help="Filename for saving result")
    parser.add_argument("-L", "--save_label", action="store_true", help="Also save label data, useful for plot only")
    parser.add_argument("-E", "--save_embeddings", action="store_true", help="Also save embeddings data")
    parser.add_argument("-B", "--bunch", action="store_true", help="Run all 8 tests N{0,1}D{0,1}F{0,1}")
    parser.add_argument("-F", "--force_reload", action="store_true", help="Force reload, instead of using cache")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for get_embeddings")
    parser.add_argument("-p", "--plot_only", nargs="*", type=str, help="Plot saved results, Format 1 2 3 or 1, 2, 3 or *.npy")
    args = parser.parse_known_args(argv)[0]

    if args.plot_only != None and len(args.plot_only) != 0:
        # Plot only
        from glob2 import glob

        score_files = []
        for ss in args.plot_only:
            score_files.extend(glob(ss.replace(",", "").strip()))
        args.plot_only = score_files
    elif args.model_file == None:
        print("Please provide -m MODEL_FILE")
        exit(1)
    else:
        if args.model_file.endswith(".h5"):
            # Keras model file "model.h5"
            model_name = os.path.splitext(os.path.basename(args.model_file))[0]
        else:
            # MXNet model file "models/r50-arcface-emore/model,1"
            model_name = os.path.basename(os.path.dirname(args.model_file))

        if args.save_result == default_save_result_name:
            args.save_result = default_save_result_name.format(model_name=model_name, subset=args.subset)
    return args


if __name__ == "__main__":
    import sys

    args = parse_arguments(sys.argv[1:])
    if args.plot_only != None and len(args.plot_only) != 0:
        plot_roc_and_calculate_tpr(args.plot_only)
    else:
        if args.model_file.endswith(".h5"):
            import tensorflow as tf
            interf_func = keras_model_interf(args.model_file)
        else:
            import mxnet as mx
            interf_func = Mxnet_model_interf(args.model_file)
        save_name = os.path.splitext(args.save_result)[0]
        if args.bunch:
            results, embs, embs_f, templates, medias, p1, p2, label, face_scores = run_model_test_bunch(
                args.data_path, args.subset, interf_func, batch_size=args.batch_size, force_reload=args.force_reload
            )
            scores = list(results.values())
            names = [save_name + "_" + ii for ii in results.keys()]
        else:
            score, embs, embs_f, templates, medias, p1, p2, label, face_scores = run_model_test(
                args.data_path, args.subset, interf_func, batch_size=args.batch_size, force_reload=args.force_reload
            )
            scores, names = [score], [save_name]

        if args.save_embeddings:
            np.savez(args.save_result, scores=scores, names=names, embs=embs, embs_f=embs_f, label=label)
        elif args.save_label:
            np.savez(args.save_result, scores=scores, names=names, label=label)
        else:
            np.savez(args.save_result, scores=scores, names=names)
        plot_roc_and_calculate_tpr(scores, names=names, label=label)
else:
    try:
        import mxnet as mx
        import tensorflow as tf
    except:
        pass
