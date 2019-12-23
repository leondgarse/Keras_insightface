# ___Advanced Implementation___
***

## Introduce
  - In this advanced implementation, we will realize:
    - Evaluating callbacks using bin files
    - A general model used for all loss train
    - Center loss and combining with softmax and arcface
## Default import
  - These import will be default
    ```py
    import os
    import sys
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    ```
## Loading data by Datasets
  - We will use `bin files` for evaluation, so the whole training data can be used for training
    ```py
    ''' get image paths from data folder '''
    import glob2
    import pickle
    image_names = glob2.glob('/datasets/faces_emore_112x112_folders/*/*.jpg')
    image_names = np.random.permutation(image_names).tolist()
    image_classes = [int(os.path.basename(os.path.dirname(ii))) for ii in image_names]

    ''' Backup to pickle '''
    with open('faces_emore_img_class_shuffle.pkl', 'wb') as ff:
        pickle.dump({'image_names': image_names, "image_classes": image_classes}, ff)
    ```
    Next time, we can just restore those image paths from pickle.
    ```py
    ''' Restore from pickle '''
    import pickle
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    with open('faces_emore_img_class_shuffle.pkl', 'rb') as ff:
        aa = pickle.load(ff)
    image_names, image_classes = aa['image_names'], aa['image_classes']
    classes = np.max(image_classes) + 1
    print(len(image_names), len(image_classes), classes)
    # 5822653 5822653 85742

    ''' Construct a dataset by image_names '''
    # list_ds = tf.data.Dataset.list_files('/datasets/faces_emore_112x112_folders/*/*')
    list_ds = tf.data.Dataset.from_tensor_slices(image_names)

    ''' Construct a dataset generating images and one-hot labels '''
    def process_path(file_path, classes, img_shape=(112, 112)):
        parts = tf.strings.split(file_path, os.path.sep)[-2]
        label = tf.cast(tf.strings.to_number(parts), tf.int32)
        label = tf.one_hot(label, depth=85742, dtype=tf.int32)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, img_shape)
        img = tf.image.random_flip_left_right(img)
        return img, label

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=None, batch_size=128):
        if cache:
            ds = ds.cache(cache) if isinstance(cache, str) else ds.cache()
        if shuffle_buffer_size == None:
            shuffle_buffer_size = batch_size * 100

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat()
        ds = ds.map(lambda xx: process_path(xx, classes), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    # train_ds = prepare_for_training(labeled_ds, cache="/tmp/faces_emore.tfcache")
    batch_size = 128 * len(tf.config.experimental.get_visible_devices('GPU'))
    train_ds = prepare_for_training(list_ds, cache=False, shuffle_buffer_size=len(image_names), batch_size=batch_size)
    steps_per_epoch = np.ceil(len(image_names) / batch_size)

    image_batch, label_batch = next(iter(train_ds))
    print(image_batch.shape, label_batch.shape)
    # (128, 112, 112, 3) (128, 85742)
    ```
## Evaluation callback
  - **Evaluation** is triggered on epoch end, so we can use custom callbacks to realize this
  - **Bin files** includes `image data bins` and `is same labels`. We can measure if two images is same person or not by their distance
    - **Distance** can be measured by `cosine` or `pairwise` or others. Here we will use `cosine` distance.
    - **threshold** is a number, that any distance greater (cosine distance) / less (pairwise distance) than this value is judged as a same person
  - **Python code** implement a sub class of `tf.keras.callbacks.Callback`, re-write `on_epoch_end` function to do evaluation, and can also save model
    ```py
    import pickle
    import io
    from tqdm import tqdm
    from skimage.io import imread
    from sklearn.preprocessing import normalize

    class epoch_eval_callback(tf.keras.callbacks.Callback):
        def __init__(self, test_bin_file, batch_size=128, rescale=1./255, save_model=None):
            super(epoch_eval_callback, self).__init__()
            bins, issame_list = np.load(test_bin_file, encoding='bytes', allow_pickle=True)
            ds = tf.data.Dataset.from_tensor_slices(bins)
            _imread = lambda xx: tf.image.convert_image_dtype(tf.image.decode_jpeg(xx), dtype=tf.float32)
            ds = ds.map(_imread)
            self.ds = ds.batch(128)
            self.test_issame = np.array(issame_list)
            self.test_names = os.path.splitext(os.path.basename(test_bin_file))[0]
            self.max_accuracy = 0
            self.batch_size = batch_size
            self.steps = int(np.ceil(len(bins) / batch_size))
            self.save_model = save_model

        # def on_batch_end(self, batch=0, logs=None):
        def on_epoch_end(self, epoch=0, logs=None):
            dists = []
            embs = []
            tf.print("\n")
            for img_batch in tqdm(self.ds, 'Evaluating ' + self.test_names, total=self.steps):
                emb = basic_model.predict(img_batch)
                embs.extend(emb)
            embs = np.array(embs)
            if np.isnan(embs).sum() != 0:
                tf.print("NAN in embs, not a good one")
                return
            embs = normalize(embs)
            embs_a = embs[::2]
            embs_b = embs[1::2]
            dists = (embs_a * embs_b).sum(1)

            self.tt = np.sort(dists[self.test_issame[:dists.shape[0]]])
            self.ff = np.sort(dists[np.logical_not(self.test_issame[:dists.shape[0]])])

            max_accuracy = 0
            thresh = 0
            for vv in reversed(self.ff[-300:]):
                acc_count = (self.tt > vv).sum() + (self.ff <= vv).sum()
                acc = acc_count / dists.shape[0]
                if acc > max_accuracy:
                    max_accuracy = acc
                    thresh = vv
            tf.print("\n")
            if max_accuracy > self.max_accuracy:
                is_improved = True
                self.max_accuracy = max_accuracy
                if self.save_model:
                    save_path = '%s_%d' % (self.save_model, epoch)
                    tf.print("Saving model to: %s" % (save_path))
                    model.save(save_path)
            else:
                is_improved = False
            tf.print(">>>> %s evaluation max accuracy: %f, thresh: %f, overall max accuracy: %f, improved = %s" % (self.test_names, max_accuracy, thresh, self.max_accuracy, is_improved))
    ```
  - **Test** We can use the `basic model` or any other model outputs `embeddings` to test this function
    ```py
    basic_model = ...
    basic_model.load_weights('basi_model_softmax.h5')

    aa = epoch_eval_callback('/datasets/faces_emore/lfw.bin', save_model='./test')
    aa.on_epoch_end()
    ```
## General model
  - Model definition including a basic model outputs `embeddings` and a `logits` bottleneck layer used for training classification
    - **Basic model** output is used for `center loss` and `triplet loss` training
    - **Bottleneck layer** is a `WeightNormalization` dense layer
    - **Output** of this model is the concatenate of `embeddings` and `logits`
  - **Python code**
    ```py
    from tensorflow.keras import layers

    ''' Basic model '''
    # xx = keras.applications.ResNet101V2(include_top=False, weights='imagenet')
    # xx = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
    # xx = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
    xx = tf.keras.applications.ResNet50V2(input_shape=(112, 112, 3), include_top=False, weights='imagenet')
    xx.trainable = True

    inputs = xx.inputs[0]
    nn = xx.outputs[0]
    nn = layers.GlobalAveragePooling2D()(nn)
    nn = layers.Dropout(0.1)(nn)
    embedding = layers.Dense(512, name='embedding')(nn)
    basic_model = keras.models.Model(inputs, embedding)

    ''' Model with bottleneck '''
    class NormDense(tf.keras.layers.Layer):
        def __init__(self, classes=1000, **kwargs):
            super(NormDense, self).__init__(**kwargs)
            self.output_dim = classes
        def build(self, input_shape):
            self.w = self.add_weight(name='norm_dense_w', shape=(input_shape[-1], self.output_dim), initializer='random_normal', trainable=True)
            super(NormDense, self).build(input_shape)
        def call(self, inputs, **kwargs):
            norm_w = tf.nn.l2_normalize(self.w, axis=0)
            # inputs = tf.nn.l2_normalize(inputs, axis=1)
            return tf.matmul(inputs, norm_w)
        def compute_output_shape(self, input_shape):
            shape = tf.TensorShape(input_shape).as_list()
            shape[-1] = self.output_dim
            return tf.TensorShape(shape)
        def get_config(self):
            base_config = super(NormDense, self).get_config()
            base_config['output_dim'] = self.output_dim
        @classmethod
        def from_config(cls, config):
            return cls(**config)

    inputs = basic_model.inputs[0]
    embedding = basic_model.outputs[0]
    output = NormDense(classes, name='norm_dense')(embedding)
    concate = layers.concatenate([embedding, output], name='concate')
    model = keras.models.Model(inputs, concate)
    # model.load_weights('nn.h5')
    model.summary()

    ''' logits accuracy '''
    def logits_accuracy(y_true, y_pred):
        logits = y_pred[:, 512:]
        return keras.metrics.categorical_accuracy(y_true, logits)
    ```
## Callbacks
  - **Callbacks** includes
    - **ModelCheckpoint** saves model weights on every epoch end
    - **ReduceLROnPlateau** adjusts learning rate on every epoch end
    - **Evaluation functions** evaluate model accuracy on every epoch end
    - Evaluation functions can also be set to saving model weights on epoch end according to the evaluating accuracy
  - **Python code**
    ```py
    ''' Callbacks '''
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

    # lfw_eval = epoch_eval_callback('/datasets/faces_emore/lfw.bin', save_model="keras_model_lfw_acc.h5")
    lfw_eval = epoch_eval_callback('/datasets/faces_emore/lfw.bin', save_model=None)
    agedb_30_eval = epoch_eval_callback('/datasets/faces_emore/agedb_30.bin', save_model=None)
    cfp_fp_eval = epoch_eval_callback('/datasets/faces_emore/cfp_fp.bin', save_model=None)

    # reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=5, verbose=1)
    def scheduler(epoch):
        lr = 0.001 if epoch < 10 else 0.001 * np.exp(0.2 * (10 - epoch))
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, lr))
        return lr
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model_checkpoint = ModelCheckpoint("./keras_checkpoints.h5", verbose=1)
    # model_checkpoint = ModelCheckpoint("./keras_checkpoints_res_arcface", 'val_loss', verbose=1, save_best_only=True)
    callbacks = [lr_scheduler, lfw_eval, agedb_30_eval, cfp_fp_eval, model_checkpoint]
    ```
## Softmax loss train
  - Softmax loss training uses `logits` only
    ```py
    def softmax_loss(y_true, y_pred):
        logits = y_pred[:, 512:]
        return keras.losses.categorical_crossentropy(y_true, logits, from_logits=True)

    model.compile(optimizer='adamax', loss=softmax_loss, metrics=[logits_accuracy])
    hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, initial_epoch=0)
    ```
## Arcface loss train
  - Arcface loss training uses both `embeddings` and `logits`
    - Here `embeddings` is used to normalize `logits`
    - We cannot just use `l2_normalize` over `logits`. It will mix all these parameters together, making it hard to decrease loss value
  - **Python code**
    ```py
    # def arcface_loss(y_true, y_pred, margin1=1.0, margin2=0.2, margin3=0.3, scale=64.0):
    def arcface_loss(y_true, y_pred, margin1=0.9, margin2=0.4, margin3=0.15, scale=64.0):
        embedding = y_pred[:, :512]
        logits = y_pred[:, 512:]
        norm_emb = tf.norm(embedding, axis=1, keepdims=True)
        norm_logits = logits / norm_emb

        theta = tf.acos(norm_logits)
        cond = tf.where(tf.greater(theta * margin1 + margin3, np.pi), tf.zeros_like(y_true), y_true)
        cond = tf.cast(cond, dtype=tf.bool)
        m1_theta_plus_m3 = tf.where(cond, theta * margin1 + margin3, theta)
        cos_m1_theta_plus_m3 = tf.cos(m1_theta_plus_m3)
        arcface_logits = tf.where(cond, cos_m1_theta_plus_m3 - margin2, cos_m1_theta_plus_m3) * scale
        # tf.assert_equal(tf.math.is_nan(tf.reduce_mean(arcface_logits)), False)
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=True)

    model.compile(optimizer='adamax', loss=arcface_loss, metrics=[logits_accuracy])
    hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, initial_epoch=0)
    ```
## Center loss
  - **Center loss training** uses a `center` and `embedding` value
    - **center** needs to be saved for re-train and fine-tune, so we need a `call back` here
    - Center loss training should be better combined with other loss types
  - **Python code of center loss**
    ```py
    ''' Callback to save center values on every epoch end '''
    class Save_Numpy_Callback(tf.keras.callbacks.Callback):
        def __init__(self, save_file, save_tensor):
            super(Save_Numpy_Callback, self).__init__()
            self.save_file = os.path.splitext(save_file)[0]
            self.save_tensor = save_tensor

        def on_epoch_end(self, epoch=0, logs=None):
            np.save(self.save_file, self.save_tensor.numpy())

    ''' Center loss class '''
    class Center_loss(keras.losses.Loss):
        def __init__(self, num_classes, feature_dim=512, alpha=0.5, factor=1.0, initial_file=None):
            super(Center_loss, self).__init__()
            self.alpha = alpha
            self.factor = factor
            centers = tf.Variable(tf.zeros([num_classes, feature_dim]))
            if initial_file:
                if os.path.exists(initial_file):
                    aa = np.load(initial_file)
                    centers.assign(aa)
                self.save_centers_callback = Save_Numpy_Callback(initial_file, centers)
            self.centers = centers

        def call(self, y_true, y_pred):
            labels = tf.argmax(y_true, axis=1)
            centers_batch = tf.gather(self.centers, labels)
            # loss = tf.reduce_mean(tf.square(y_pred - centers_batch))
            loss = tf.reduce_mean(tf.square(y_pred - centers_batch), axis=-1)

            # Update centers
            # diff = (1 - self.alpha) * (centers_batch - y_pred)
            diff = centers_batch - y_pred
            unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])

            diff = diff / tf.cast((1 + appear_times), tf.float32)
            diff = self.alpha * diff
            # tf.print(centers_batch.shape, self.centers.shape, labels.shape, diff.shape)
            self.centers.assign(tf.tensor_scatter_nd_sub(self.centers, tf.expand_dims(labels, 1), diff))
            # centers_batch = tf.gather(self.centers, labels)

            return loss * self.factor

    ''' Define center_loss, append its callback to other callbacks '''
    center_loss = Center_loss(classes, factor=1.0, initial_file='./centers.npy')
    callbacks.append(center_loss.save_centers_callback)
    ```
  - **Combined center loss** with other loss types
    ```py
    def center_loss_wrapper(center_loss, other_loss):
        def _loss_func(y_true, y_pred):
            embedding = y_pred[:, :512]
            center_loss_v = center_loss(y_true, embedding)
            other_loss_v = other_loss(y_true, y_pred)
            # tf.print("other_loss: %s, cent_loss: %s" % (other_loss_v, center_loss_v))
            return other_loss_v + center_loss_v
        other_loss_name = other_loss.name if 'name' in other_loss.__dict__ else other_loss.__name__
        _loss_func.__name__ = "center_" + other_loss_name
        return _loss_func

    center_softmax_loss = center_loss_wrapper(center_loss, softmax_loss)
    center_arcface_loss = center_loss_wrapper(center_loss, arcface_loss)

    def single_center_loss(labels, prediction):
        embedding = prediction[:, :512]
        norm_logits = prediction[:, 512:]
        return center_loss(labels, embedding)

    # model.compile(optimizer='adamax', loss=single_center_loss, metrics=[logits_accuracy()])
    model.compile(optimizer='adamax', loss=center_softmax_loss, metrics=[logits_accuracy()])
    hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, initial_epoch=0)
    ```
  - **Training process**
    ```py
    Epoch 1/200
    >>>> lfw evaluation max accuracy: 0.956833, thresh: 0.628311, overall max accuracy: 0.956833, improved = True
    43216/43216 [==============================] - 9838s 228ms/step - loss: 9.3089 - logits_accuracy: 0.0376 - val_loss: 7.7020 - val_logits_accuracy: 0.1513
    Epoch 2/200
    >>>> lfw evaluation max accuracy: 0.986000, thresh: 0.321373, overall max accuracy: 0.986000, improved = True
    43216/43216 [==============================] - 9979s 231ms/step - loss: 6.3202 - logits_accuracy: 0.4252 - val_loss: 5.1966 - val_logits_accuracy: 0.6057
    Epoch 3/200
    >>>> lfw evaluation max accuracy: 0.991667, thresh: 0.287180, overall max accuracy: 0.991667, improved = True
    43216/43216 [==============================] - 9476s 219ms/step - loss: 4.5633 - logits_accuracy: 0.7169 - val_loss: 3.9777 - val_logits_accuracy: 0.7618
    Epoch 4/200
    >>>> lfw evaluation max accuracy: 0.992333, thresh: 0.250578, overall max accuracy: 0.992333, improved = True
    43216/43216 [==============================] - 9422s 218ms/step - loss: 3.6551 - logits_accuracy: 0.8149 - val_loss: 3.2682 - val_logits_accuracy: 0.8270
    Epoch 5/200
    >>>> lfw evaluation max accuracy: 0.993500, thresh: 0.232111, overall max accuracy: 0.993500, improved = True
    43216/43216 [==============================] - 9379s 217ms/step - loss: 3.1123 - logits_accuracy: 0.8596 - val_loss: 2.8836 - val_logits_accuracy: 0.8516
    Epoch 6/200
    >>>> lfw evaluation max accuracy: 0.992500, thresh: 0.208816, overall max accuracy: 0.993500, improved = False
    43216/43216 [==============================] - 9068s 210ms/step - loss: 2.7492 - logits_accuracy: 0.8851 - val_loss: 2.5630 - val_logits_accuracy: 0.8771
    Epoch 7/200
    >>>> lfw evaluation max accuracy: 0.992667, thresh: 0.207485, overall max accuracy: 0.993500, improved = False
    43216/43216 [==============================] - 9145s 212ms/step - loss: 2.4826 - logits_accuracy: 0.9015 - val_loss: 2.3668 - val_logits_accuracy: 0.8881
    ```
***
