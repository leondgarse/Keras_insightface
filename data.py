import os
import glob2
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.io import imread

# /datasets/faces_emore_112x112_folders/*/*.jpg'
default_image_names_reg = "*/*.jpg"
default_image_classes_rule = lambda path: int(os.path.basename(os.path.dirname(path)))


def pre_process_folder(data_path, image_names_reg=None, image_classes_rule=None):
    while data_path.endswith("/"):
        data_path = data_path[:-1]
    if not data_path.endswith(".npz"):
        dest_pickle = os.path.join("./", os.path.basename(data_path) + "_shuffle.npz")
    else:
        dest_pickle = data_path

    if os.path.exists(dest_pickle):
        aa = np.load(dest_pickle)
        if len(aa.keys()) == 2:
            image_names, image_classes, embeddings = aa["image_names"], aa["image_classes"], []
        else:
            # dataset with embedding values
            image_names, image_classes, embeddings = aa["image_names"], aa["image_classes"], aa["embeddings"]
        print(">>>> reloaded from dataset backup:", dest_pickle)
    else:
        if not os.path.exists(data_path):
            return [], [], [], 0, None
        if image_names_reg is None or image_classes_rule is None:
            image_names_reg, image_classes_rule = default_image_names_reg, default_image_classes_rule
        image_names = glob2.glob(os.path.join(data_path, image_names_reg))
        image_names = np.random.permutation(image_names).tolist()
        image_classes = [image_classes_rule(ii) for ii in image_names]
        embeddings = np.array([])
        np.savez_compressed(dest_pickle, image_names=image_names, image_classes=image_classes)
    classes = np.max(image_classes) + 1
    return image_names, image_classes, embeddings, classes, dest_pickle


def tf_imread(file_path):
    # tf.print('Reading file:', file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)  # [0, 255]
    img = tf.cast(img, "float32")  # [0, 255]
    return img


def random_process_image(img, img_shape=(112, 112), random_status=2, random_crop=None):
    if random_status >= 0:
        img = tf.image.random_flip_left_right(img)
    if random_status >= 1:
        # 25.5 == 255 * 0.1
        img = tf.image.random_brightness(img, 25.5 * random_status)
    if random_status >= 2:
        img = tf.image.random_contrast(img, 1 - 0.1 * random_status, 1 + 0.1 * random_status)
        img = tf.image.random_saturation(img, 1 - 0.1 * random_status, 1 + 0.1 * random_status)
    if random_status >= 3 and random_crop is not None:
        img = tf.image.random_crop(img, random_crop)
        img = tf.image.resize(img, img_shape)

    if random_status >= 1:
        img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def pick_by_image_per_class(image_classes, image_per_class):
    cc = pd.value_counts(image_classes)
    class_pick = cc[cc >= image_per_class].index
    return np.array([ii in class_pick for ii in image_classes]), class_pick


def prepare_dataset(
    data_path,
    image_names_reg=None,
    image_classes_rule=None,
    batch_size=128,
    img_shape=(112, 112),
    random_status=2,
    random_crop=(100, 100, 3),
    image_per_class=0,
    cache=False,
    shuffle_buffer_size=None,
    is_train=True,
    teacher_model_interf=None,
):
    image_names, image_classes, embeddings, classes, _ = pre_process_folder(data_path, image_names_reg, image_classes_rule)
    if len(image_names) == 0:
        return None
    print(">>>> Image length: %d, Image class length: %d, classes: %d" % (len(image_names), len(image_classes), classes))
    if image_per_class != 0:
        pick, class_pick = pick_by_image_per_class(image_classes, image_per_class)
        image_names, image_classes = image_names[pick], image_classes[pick]
        if len(embeddings) != 0:
            embeddings = embeddings[pick]
        print(">>>> After pick[%d], images: %d, valid classes: %d" % (image_per_class, len(image_names), class_pick.shape[0]))

    if len(embeddings) != 0 and teacher_model_interf is None:
        # dataset with embedding values
        print(">>>> embeddings: %s. This takes some time..." % (np.shape(embeddings),))
        ds = tf.data.Dataset.from_tensor_slices((image_names, embeddings, image_classes))
        process_func = lambda imm, emb, label: (tf_imread(imm), (emb, tf.one_hot(label, depth=classes, dtype=tf.int32)))
    else:
        ds = tf.data.Dataset.from_tensor_slices((image_names, image_classes))
        process_func = lambda imm, label: (tf_imread(imm), tf.one_hot(label, depth=classes, dtype=tf.int32))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.shuffle(buffer_size=len(image_names))
    ds = ds.map(process_func, num_parallel_calls=AUTOTUNE)

    if is_train and random_status >= 0:
        random_process_func = lambda xx, yy: (random_process_image(xx, img_shape, random_status, random_crop), yy)
        ds = ds.map(random_process_func, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)  # Use batch --> map has slightly effect on dataset reading time, but harm the randomness
    if teacher_model_interf is not None:
        print(">>>> Teacher model interface provided.")
        emb_func = lambda imm, label: (imm, (teacher_model_interf(imm), label))
        ds = ds.map(emb_func, num_parallel_calls=AUTOTUNE)

    ds = ds.map(lambda xx, yy: ((xx - 127.5) * 0.0078125, yy))
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    steps_per_epoch = int(np.ceil(len(image_names) / float(batch_size)))
    return ds, steps_per_epoch


def prepare_distill_dataset_tfrecord(data_path, batch_size=128, img_shape=(112, 112), random_status=2, random_crop=(100, 100, 3), **kw):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    decode_base_info = {
        "classes": tf.io.FixedLenFeature([], dtype=tf.int64),
        "emb_shape": tf.io.FixedLenFeature([], dtype=tf.int64),
        "total": tf.io.FixedLenFeature([], dtype=tf.int64),
        "use_fp16": tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    decode_feature = {
        "image_names": tf.io.FixedLenFeature([], dtype=tf.string),
        "image_classes": tf.io.FixedLenFeature([], dtype=tf.int64),
        # "embeddings": tf.io.FixedLenFeature([emb_shape], dtype=tf.float32),
        "embeddings": tf.io.FixedLenFeature([], dtype=tf.string),
    }

    # base info saved in the first data line
    header = tf.data.TFRecordDataset([data_path]).as_numpy_iterator().next()
    hh = tf.io.parse_single_example(header, decode_base_info)
    classes, emb_shape, total, use_fp16 = hh["classes"].numpy(), hh["emb_shape"].numpy(), hh["total"].numpy(), hh["use_fp16"].numpy()
    emb_dtype = tf.float16 if use_fp16 else tf.float32
    print(">>>> [Base info] total:", total, "classes:", classes, "emb_shape:", emb_shape, "use_fp16:", use_fp16)

    def decode_fn(record_bytes):
        ff = tf.io.parse_single_example(record_bytes, decode_feature)
        image_name, image_classe, embedding = ff["image_names"], ff["image_classes"], ff["embeddings"]
        img = random_process_image(tf_imread(image_name), img_shape, random_status, random_crop)
        label = tf.one_hot(image_classe, depth=classes, dtype=tf.int32)
        embedding = tf.io.decode_raw(embedding, emb_dtype)
        embedding.set_shape([emb_shape])
        return img, (embedding, label)

    ds = tf.data.TFRecordDataset([data_path])
    ds = ds.shuffle(buffer_size=batch_size * 1000)
    ds = ds.map(decode_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda xx, yy: ((xx - 127.5) * 0.0078125, yy))
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    steps_per_epoch = int(np.ceil(total / float(batch_size)))
    return ds, steps_per_epoch


class Triplet_dataset:
    def __init__(
        self,
        data_path,
        image_names_reg=None,
        image_classes_rule=None,
        batch_size=48,
        image_per_class=4,
        img_shape=(112, 112, 3),
        random_status=3,
        random_crop=(100, 100, 3),
        teacher_model_interf=None,
    ):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        image_names, image_classes, embeddings, classes, _ = pre_process_folder(data_path, image_names_reg, image_classes_rule)
        image_per_class = max(4, image_per_class)
        pick, _ = pick_by_image_per_class(image_classes, image_per_class)
        image_names, image_classes = image_names[pick].astype(str), image_classes[pick]
        self.classes = classes

        image_dataframe = pd.DataFrame({"image_names": image_names, "image_classes": image_classes})
        self.image_dataframe = image_dataframe.groupby("image_classes").apply(lambda xx: xx.image_names.values)
        self.split_func = lambda xx: np.array(
            np.split(np.random.permutation(xx)[: len(xx) // image_per_class * image_per_class], len(xx) // image_per_class)
        )
        self.image_per_class = image_per_class
        self.batch_size = batch_size // image_per_class * image_per_class
        self.img_shape = img_shape[:2]
        self.channels = img_shape[2] if len(img_shape) > 2 else 3
        print("The final train_dataset batch will be %s" % ([self.batch_size, *self.img_shape, self.channels]))

        one_hot_label = lambda label: tf.one_hot(label, depth=classes, dtype=tf.int32)
        random_imread = lambda imm: random_process_image(tf_imread(imm), self.img_shape, random_status, random_crop)
        if len(embeddings) != 0 and teacher_model_interf is None:
            self.teacher_embeddings = dict(zip(image_names, embeddings[pick]))
            emb_spec = tf.TensorSpec(shape=(embeddings.shape[-1],), dtype=tf.float32)
            output_signature = (tf.TensorSpec(shape=(), dtype=tf.string), emb_spec, tf.TensorSpec(shape=(), dtype=tf.int64))
            ds = tf.data.Dataset.from_generator(self.image_shuffle_gen_with_emb, output_signature=output_signature)
            process_func = lambda imm, emb, label: (random_imread(imm), (emb, one_hot_label(label)))
        else:
            output_signature = (tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int64))
            ds = tf.data.Dataset.from_generator(self.image_shuffle_gen, output_signature=output_signature)
            process_func = lambda imm, label: (random_imread(imm), one_hot_label(label))
        ds = ds.map(process_func, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(self.batch_size)
        if teacher_model_interf is not None:
            print(">>>> Teacher model interference provided.")
            emb_func = lambda imm, label: (imm, (teacher_model_interf(imm), label))
            ds = ds.map(emb_func, num_parallel_calls=AUTOTUNE)

        ds = ds.map(lambda xx, yy: ((xx - 127.5) * 0.0078125, yy))
        self.ds = ds.prefetch(buffer_size=AUTOTUNE)

        shuffle_dataset = self.image_dataframe.map(self.split_func)
        self.total = np.vstack(shuffle_dataset.values).flatten().shape[0]
        self.steps_per_epoch = int(np.ceil(self.total / float(batch_size)))

    def image_shuffle_gen(self):
        tf.print("Shuffle image data...")
        shuffle_dataset = self.image_dataframe.map(self.split_func)
        image_data = np.random.permutation(np.vstack(shuffle_dataset.values)).flatten()
        return ((ii, int(ii.split(os.path.sep)[-2])) for ii in image_data)

    def image_shuffle_gen_with_emb(self):
        tf.print("Shuffle image with embedding data...")
        shuffle_dataset = self.image_dataframe.map(self.split_func)
        image_data = np.random.permutation(np.vstack(shuffle_dataset.values)).flatten()
        return ((ii, self.teacher_embeddings[ii], int(ii.split(os.path.sep)[-2])) for ii in image_data)
