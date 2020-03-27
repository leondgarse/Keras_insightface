import os
import glob2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# /datasets/faces_emore_112x112_folders/*/*.jpg'
def pre_process_folder(data_path):
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
        image_names = glob2.glob(os.path.join(data_path, "*/*.jpg"))
        image_names = np.random.permutation(image_names).tolist()
        image_classes = [int(os.path.basename(os.path.dirname(ii))) for ii in image_names]
        with open(dest_pickle, "wb") as ff:
            pickle.dump({"image_names": image_names, "image_classes": image_classes}, ff)
    classes = np.max(image_classes) + 1
    return image_names, image_classes, classes


def process_path(file_path, classes=0, img_shape=(112, 112), random_status=2, one_hot_label=True):
    parts = tf.strings.split(file_path, os.path.sep)[-2]
    label = tf.cast(tf.strings.to_number(parts), tf.int32)
    if one_hot_label:
        label = tf.one_hot(label, depth=classes, dtype=tf.int32)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if random_status >= 1:
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_flip_left_right(img)
    if random_status >= 2:
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
    if random_status >= 3:
        img = tf.image.random_crop(img, [100, 100, 3])
        img = tf.image.resize(img, img_shape)
    img = (img - 0.5) * 2
    return img, label


def prepare_for_training(data_path, batch_size=128, random_status=2, cache=True, shuffle_buffer_size=None):
    image_names, image_classes, classes = pre_process_folder(data_path)
    if len(image_names) == 0:
        return None, 0
    print(len(image_names), len(image_classes), classes)

    ds = tf.data.Dataset.from_tensor_slices(image_names)
    # batch_size = batch_size * len(tf.config.experimental.get_visible_devices("GPU"))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if shuffle_buffer_size == None:
        shuffle_buffer_size = batch_size * 100

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    if cache:
        ds = ds.cache(cache) if isinstance(cache, str) else ds.cache()
    ds = ds.repeat()
    ds = ds.map(lambda xx: process_path(xx, classes, random_status=random_status), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    steps_per_epoch = np.ceil(len(image_names) / batch_size)
    return ds, steps_per_epoch, classes


class Triplet_dataset:
    def __init__(self, data_path, batch_size=48, image_per_class=4, img_shape=(112, 112, 3), random_status=3):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        image_names, image_classes, _ = pre_process_folder(data_path)
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
            img_name, img_shape=self.img_shape, random_status=random_status, one_hot_label=False
        )
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
        images, labels = tf.map_fn(self.process_path, image_names, dtype=(tf.float32, tf.int32))
        return images, labels
