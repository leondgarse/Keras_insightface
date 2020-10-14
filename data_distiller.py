#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import normalize
from data import pre_process_folder, tf_imread

try:
    import mxnet as mx
except:
    pass


def get_mxnet_model(model, layer="fc1", image_size=(112, 112)):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if len(cvd) > 0 and int(cvd) != -1:
        ctx = [mx.gpu(ii) for ii in range(len(cvd.split(",")))]
    else:
        ctx = [mx.cpu()]

    prefix, epoch = model.split(",")
    print(">>>> loading mxnet model:", prefix, epoch, ctx)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, int(epoch))
    all_layers = sym.get_internals()
    sym = all_layers[layer + "_output"]
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[("data", (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


def mxnet_model_infer(model, imgs):
    imgs = imgs.transpose(0, 3, 1, 2)
    data = mx.nd.array(imgs)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    emb = model.get_outputs()[0].asnumpy()
    return emb


def data_distiller(data_path, model, dest_file=None, batch_size=256, limit=-1):
    """ Init dataset """
    image_names, image_classes, _, _, dataset_pickle_file_src = pre_process_folder(data_path)
    if limit != -1:
        image_names, image_classes = image_names[:limit], image_classes[:limit]

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((image_names, image_classes))
    ds = ds.batch(batch_size)
    total = int(np.ceil(len(image_names) // batch_size)) + 1

    """ Init model, it could be TF model / MXNet model file / keras model file """
    if isinstance(model, str):
        if model.endswith(".h5"):
            # Keras model file
            basic_model = tf.keras.models.load_model(model, compile=False)
            infer = lambda imgs: basic_model((imgs * 2) - 1).numpy()
        else:
            # MXNet model file, like models/r50-arcface-emore/model,1
            basic_model = get_mxnet_model(model)
            infer = lambda imgs: mxnet_model_infer(basic_model, imgs.numpy() * 255)
    else:
        # TF model
        infer = lambda imgs: model((imgs * 2) - 1).numpy()

    """ Extract embeddings """
    new_image_names, new_image_classes, embeddings = [], [], []
    for imm, label in tqdm(ds, "Embedding", total=total):
        imgs = tf.stack([tf_imread(ii) for ii in imm])
        emb = normalize(infer(imgs), axis=1)

        new_image_names.extend(imm.numpy())
        new_image_classes.extend(label.numpy())
        embeddings.extend(emb)
    # imms, labels, embeddings = np.array(imms), np.array(labels), np.array(embeddings)

    """ Save to npz """
    if dest_file is None:
        src_name = os.path.splitext(os.path.basename(dataset_pickle_file_src))[0]
        dest_file = src_name + "_label_embs_normed_{}.npz".format(embeddings[0].shape[0])
    np.savez_compressed(dest_file, image_names=new_image_names, image_classes=new_image_classes, embeddings=embeddings)
    # with open(dest_file, "wb") as ff:
    #     pickle.dump({"image_names": new_image_names, "image_classes": new_image_classes, "embeddings": embeddings}, ff)

    return dest_file


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-M", "--model_file", type=str, required=True, help="Saved basic_model file path, NOT model, could be keras / mxnet one"
    )
    parser.add_argument("-D", "--data_path", type=str, required=True, help="Original dataset path")
    parser.add_argument("-d", "--dest_file", type=str, default=None, help="Dest file path to save the processed dataset npz")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("-L", "--limit", type=int, default=0, help="Test parameter, limit converting only the first [NUM] ones")
    args = parser.parse_known_args(sys.argv[1:])[0]

    print(">>>> Output:", data_distiller(args.data_path, args.model_file, args.dest_file, args.batch_size, args.limit))

elif __name__ == "__test__":
    batch_size = 256
    limit = 20
    dest_file = None
    data_path = "/datasets/faces_casia_112x112_folders"
    # model_file = "checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random0_arc32_E1_arcT4_BS512_casia_basic_agedb_30_epoch_37_0.946667.h5"
    # model_file = '../tdFace-flask.mxnet/models/model,0'
    model_file = "../tdFace-flask.mxnet/subcenter-arcface-logs/r100-arcface-msfdrop75/model,0"
    model = get_mxnet_model(model_file)
    imgs = tf.stack([tf_imread("../tdFace-flask/test_images/11.jpg"), tf_imread("../tdFace-flask/test_images/22.jpg")])
    ees = normalize(mxnet_model_infer(model, imgs.numpy() * 255))
    np.dot(ees, ees.T)

    data_distiller(data_path, model_file, dest_file, batch_size, limit)
