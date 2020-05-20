import os
import glob2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# /datasets/faces_emore_112x112_folders/*/*.jpg'
default_image_names_reg = "*/*.jpg"
default_image_classes_rule = lambda path: int(os.path.basename(os.path.dirname(path)))


def pre_process_folder(data_path, image_names_reg=None, image_classes_rule=None):
    if not os.path.exists(data_path):
        return np.array([]), np.array([]), 0
    while data_path.endswith("/"):
        data_path = data_path[:-1]
    dest_pickle = os.path.join("./", os.path.basename(data_path) + "_shuffle.pkl")
    if os.path.exists(dest_pickle):
        with open(dest_pickle, "rb") as ff:
            aa = pickle.load(ff)
        image_names, image_classes = aa["image_names"], aa["image_classes"]
    else:
        if image_names_reg is None or image_classes_rule is None:
            image_names_reg, image_classes_rule = default_image_names_reg, default_image_classes_rule
        image_names = glob2.glob(os.path.join(data_path, image_names_reg))
        image_names = np.random.permutation(image_names).tolist()
        image_classes = [image_classes_rule(ii) for ii in image_names]
        with open(dest_pickle, "wb") as ff:
            pickle.dump({"image_names": image_names, "image_classes": image_classes}, ff)
    classes = np.max(image_classes) + 1
    return image_names, image_classes, classes


def process_path(file_path, label, classes=0, img_shape=(112, 112), random_status=2, one_hot_label=True):
    if one_hot_label:
        label = tf.one_hot(label, depth=classes, dtype=tf.int32)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.random_flip_left_right(img)
    if random_status >= 1:
        img = tf.image.random_brightness(img, 0.1 * random_status)
    if random_status >= 2:
        img = tf.image.random_contrast(img, 1 - 0.1 * random_status, 1 + 0.1 * random_status)
        img = tf.image.random_saturation(img, 1 - 0.1 * random_status, 1 + 0.1 * random_status)
    if random_status >= 3:
        img = tf.image.random_crop(img, [100, 100, 3])
        img = tf.image.resize(img, img_shape)
    img = (tf.clip_by_value(img, 0., 1.) - 0.5) * 2
    return img, label


def prepare_dataset(
    data_path,
    image_names_reg=None,
    image_classes_rule=None,
    batch_size=128,
    random_status=2,
    cache=True,
    shuffle_buffer_size=None,
    is_train=True,
):
    image_names, image_classes, classes = pre_process_folder(data_path, image_names_reg, image_classes_rule)
    if len(image_names) == 0:
        return None, 0
    print(len(image_names), len(image_classes), classes)

    ds = tf.data.Dataset.from_tensor_slices((image_names, image_classes))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if shuffle_buffer_size == None:
        shuffle_buffer_size = batch_size * 100

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    if cache:
        ds = ds.cache(cache) if isinstance(cache, str) else ds.cache()
    if is_train:
        ds = ds.repeat()
    ds = ds.map(lambda xx, yy: process_path(xx, yy, classes, random_status=random_status), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    steps_per_epoch = np.ceil(len(image_names) / batch_size)
    return ds, steps_per_epoch, classes


class Triplet_dataset:
    def __init__(
        self,
        data_path,
        image_names_reg=None,
        image_classes_rule=None,
        batch_size=48,
        image_per_class=6,
        img_shape=(112, 112, 3),
        random_status=3,
    ):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        image_names, image_classes, _ = pre_process_folder(data_path, image_names_reg, image_classes_rule)
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

        self.process_path = lambda img_name: process_path(
            img_name, label=0, img_shape=self.img_shape, random_status=random_status, one_hot_label=False
        )
        self.get_label = lambda xx: tf.cast(tf.strings.to_number(tf.strings.split(xx, os.path.sep)[-2]), tf.int32)
        image_data = self.image_data_shuffle()
        self.steps_per_epoch = np.ceil(image_data.shape[0] / self.batch_size)

        train_dataset = tf.data.Dataset.from_generator(
            self.triplet_dataset_gen, output_types=tf.string, output_shapes=(image_per_class,)
        )
        # train_dataset = train_dataset.shuffle(total)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.map(self.process_batch_path, num_parallel_calls=self.AUTOTUNE)
        self.train_dataset = train_dataset.prefetch(buffer_size=self.AUTOTUNE)

    def triplet_dataset_gen(self):
        while True:
            tf.print("Shuffle image data...")
            image_data = self.image_data_shuffle()
            for ii in image_data:
                yield ii

    def image_data_shuffle(self):
        shuffle_dataset = self.image_dataframe.map(self.split_func)
        tt = np.random.permutation(np.vstack(shuffle_dataset.values))
        return tt

    def process_batch_path(self, image_name_batch):
        image_names = tf.reshape(image_name_batch, [-1])
        if '-dev' in tf.__version__:
            images, _ = tf.map_fn(self.process_path, image_names, fn_output_signature=(tf.float32, tf.int32))
            labels = tf.map_fn(self.get_label, image_names, fn_output_signature=tf.int32)
        else:
            images = tf.map_fn(lambda xx: self.process_path(xx)[0], image_names, dtype=tf.float32)
            labels = tf.map_fn(self.get_label, image_names, dtype=tf.int32)

        return images, labels
