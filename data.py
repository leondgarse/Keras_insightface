import os
import glob2
import numpy as np
import pandas as pd
import tensorflow as tf

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
        # with open(dest_pickle, "rb") as ff:
        #     aa = pickle.load(ff)
        if len(aa.keys()) == 2:
            image_names, image_classes, embeddings = aa["image_names"], aa["image_classes"], []
        else:
            # dataset with embedding values
            image_names, image_classes, embeddings = aa["image_names"], aa["image_classes"], aa["embeddings"]
    else:
        if not os.path.exists(data_path):
            return [], [], [], 0, None
        if image_names_reg is None or image_classes_rule is None:
            image_names_reg, image_classes_rule = default_image_names_reg, default_image_classes_rule
        image_names = glob2.glob(os.path.join(data_path, image_names_reg))
        image_names = np.random.permutation(image_names).tolist()
        image_classes = [image_classes_rule(ii) for ii in image_names]
        embeddings = []
        np.savez_compressed(dest_pickle, image_names=image_names, image_classes=image_classes)
        # with open(dest_pickle, "wb") as ff:
        #     pickle.dump({"image_names": image_names, "image_classes": image_classes}, ff)
    classes = np.max(image_classes) + 1
    return image_names, image_classes, embeddings, classes, dest_pickle


def tf_imread(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)  # [0, 255]
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0, 1]
    return img


def random_process_image(img, img_shape=(112, 112), random_status=2, random_crop=None):
    if random_status >= 0:
        img = tf.image.random_flip_left_right(img)
    if random_status >= 1:
        img = tf.image.random_brightness(img, 0.1 * random_status)
    if random_status >= 2:
        img = tf.image.random_contrast(img, 1 - 0.1 * random_status, 1 + 0.1 * random_status)
        img = tf.image.random_saturation(img, 1 - 0.1 * random_status, 1 + 0.1 * random_status)
    if random_status >= 3 and random_crop is not None:
        img = tf.image.random_crop(img, random_crop)
        img = tf.image.resize(img, img_shape)

    if random_status >= 1:
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def prepare_dataset(
    data_path,
    image_names_reg=None,
    image_classes_rule=None,
    batch_size=128,
    img_shape=(112, 112),
    random_status=2,
    random_crop=None,
    cache=False,
    shuffle_buffer_size=None,
    is_train=True,
):
    image_names, image_classes, embeddings, classes, _ = pre_process_folder(data_path, image_names_reg, image_classes_rule)
    if len(image_names) == 0:
        return None
    print(">>>> Image length: %d, Image class length: %d, classes: %d" % (len(image_names), len(image_classes), classes))

    if len(embeddings) == 0:
        ds = tf.data.Dataset.from_tensor_slices((image_names, image_classes))
        process_func = lambda imm, label: (tf_imread(imm), tf.one_hot(label, depth=classes, dtype=tf.int32))
    else:
        # dataset with embedding values
        print(">>>> embeddings: %s. This takes some time..." % (np.shape(embeddings),))
        ds = tf.data.Dataset.from_tensor_slices((image_names, image_classes, embeddings))
        process_func = lambda imm, label, emb: (tf_imread(imm), (tf.one_hot(label, depth=classes, dtype=tf.int32), emb))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.shuffle(buffer_size=len(image_names))
    ds = ds.map(process_func, num_parallel_calls=AUTOTUNE)

    if is_train and random_status >= 0:
        random_process_func = lambda xx, yy: (random_process_image(xx, img_shape, random_status, random_crop), yy)
        ds = ds.map(random_process_func , num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)  # Use batch --> map has slightly effect on dataset reading time, but harm the randomness
    ds = ds.map(lambda xx, yy: ((xx * 2) - 1, yy))
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


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
        random_crop=None,
    ):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        image_names, image_classes, _, classes, _ = pre_process_folder(data_path, image_names_reg, image_classes_rule)
        image_dataframe = pd.DataFrame({"image_names": image_names, "image_classes": image_classes})
        image_dataframe = image_dataframe.groupby("image_classes").apply(lambda xx: xx.image_names.values)
        aa = image_dataframe.map(len)
        self.image_dataframe = image_dataframe[aa > image_per_class]
        self.split_func = lambda xx: np.array(
            np.split(np.random.permutation(xx)[: len(xx) // image_per_class * image_per_class], len(xx) // image_per_class)
        )
        self.image_per_class = image_per_class
        self.batch_size = batch_size
        self.img_shape = img_shape[:2]
        self.channels = img_shape[2] if len(img_shape) > 2 else 3
        print("The final train_dataset batch will be %s" % ([batch_size * image_per_class, *self.img_shape, self.channels]))

        get_label = lambda xx: tf.one_hot(
            tf.cast(tf.strings.to_number(tf.strings.split(xx, os.path.sep)[-2]), tf.int32), depth=classes, dtype=tf.int32
        )
        self.process_path = lambda img_name: (
            random_process_image(tf_imread(img_name), self.img_shape, random_status, random_crop),
            get_label(img_name),
        )
        # image_data = self.image_data_shuffle()
        # self.steps_per_epoch = np.ceil(image_data.shape[0] / self.batch_size)

        train_dataset = tf.data.Dataset.from_generator(
            self.image_data_shuffle_gen, output_types=tf.string, output_shapes=(image_per_class,)
        )
        # train_dataset = train_dataset.shuffle(total)
        train_dataset = train_dataset.batch(self.batch_size)
        if "-dev" in tf.__version__ or int(tf.__version__.split(".")[1]) > 2:
            # tf-nightly or tf >= 2.3.0
            train_dataset = train_dataset.map(self.process_batch_path_2, num_parallel_calls=self.AUTOTUNE)
        else:
            train_dataset = train_dataset.map(self.process_batch_path_1, num_parallel_calls=self.AUTOTUNE)
        self.train_dataset = train_dataset.prefetch(buffer_size=self.AUTOTUNE)
        self.classes = classes

    def image_data_shuffle_gen(self):
        tf.print("Shuffle image data...")
        shuffle_dataset = self.image_dataframe.map(self.split_func)
        image_data = np.random.permutation(np.vstack(shuffle_dataset.values))
        return (ii for ii in image_data)

    def process_batch_path_1(self, image_name_batch):
        image_names = tf.reshape(image_name_batch, [-1])
        images, labels = tf.map_fn(self.process_path, image_names, dtype=(tf.float32, tf.int32))
        return images, labels

    def process_batch_path_2(self, image_name_batch):
        image_names = tf.reshape(image_name_batch, [-1])
        images, labels = tf.map_fn(self.process_path, image_names, fn_output_signature=(tf.float32, tf.int32))
        return images, labels
