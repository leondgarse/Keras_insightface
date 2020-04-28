import os
import glob2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


def process_path(file_path, label, classes=0, aug_policy=None, one_hot_label=True):
    if one_hot_label:
        label = tf.one_hot(label, depth=classes, dtype=tf.int32)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if aug_policy != None:
        img = aug_policy(img)
    img = (img - 0.5) * 2
    return img, label

def image_aug_random(img, random_status=3):
    img = tf.image.random_contrast(img, 1 - 0.1 * random_status, 1 + 0.1 * random_status)
    img = tf.image.random_saturation(img, 1 - 0.1 * random_status, 1 + 0.1 * random_status)
    img = tf.image.random_hue(img, 0.05 * random_status)
    return img

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

    data_df = pd.DataFrame({"image_names": image_names, "image_classes": image_classes})
    data_df.image_classes = data_df.image_classes.map(str)

    if is_train:
        if random_status != -1:
            image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,
                            rotation_range=random_status * 5,
                            # width_shift_range=random_status * 0.05,
                            # height_shift_range=random_status * 0.05,
                            brightness_range=(1.0 - random_status * 0.1, 1.0 + random_status * 0.1),
                            shear_range=random_status * 5,
                            zoom_range=random_status * 0.15,
                            # preprocessing_function=lambda img: image_aug_random(img, random_status)
            )
        else:
            from autoaugment import ImageNetPolicy
            policy = ImageNetPolicy()
            policy_func = lambda img: np.array(policy(tf.keras.preprocessing.image.array_to_img(img)), dtype=np.float32)
            image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, preprocessing_function=policy_func)
    else:
        image_gen = ImageDataGenerator(rescale=1./255)

    train_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col="image_classes", class_mode='categorical', target_size=(112, 112), batch_size=batch_size, validate_filenames=False)
    classes = data_df.image_classes.unique().shape[0]
    steps_per_epoch = np.ceil(data_df.shape[0] / batch_size)

    ''' Convert to tf.data.Dataset '''
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = tf.data.Dataset.from_generator(lambda: train_data_gen, output_types=(tf.float32, tf.int32), output_shapes=([None, 112, 112, 3], [None, classes]))
    train_ds = train_ds.map(lambda xx, yy: ((xx - 0.5) * 2, yy), num_parallel_calls=AUTOTUNE)
    # train_ds = train_ds.cache()
    # if shuffle_buffer_size == None:
    #     shuffle_buffer_size = batch_size * 100
    #
    # train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size)
    if cache:
        train_ds = train_ds.cache(cache) if isinstance(cache, str) else train_ds.cache()
    # if is_train:
    #     train_ds = train_ds.repeat()
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, steps_per_epoch, classes


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
        images, _ = tf.map_fn(self.process_path, image_names, fn_output_signature=(tf.float32, tf.int32))
        labels = tf.map_fn(self.get_label, image_names, fn_output_signature=tf.int32)
        return images, labels
