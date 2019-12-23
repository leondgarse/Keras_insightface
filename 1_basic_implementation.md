# ___Basic Implementation___
***

## Introduce
  - In this basic implementation, we will realize:
    - A basic model
    - Softmax and arcface loss function
    - Training only by training dataset folder
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
## Loading data by ImageDataGenerator
  - `ImageDataGenerator` is used for reading image data and image augment, but is not very efficiency in speed.
  - **Python code** construct an `ImageDataGenerator` from data folder. Output is images data and one-hot labels in `batch`.
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
    from keras.preprocessing.image import ImageDataGenerator

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    with open('faces_emore_img_class_shuffle.pkl', 'rb') as ff:
        aa = pickle.load(ff)
    image_names, image_classes = aa['image_names'], aa['image_classes']
    print(len(image_names), len(image_classes))
    # 5822653 5822653

    ''' Construct a dataframe feed to ImageDataGenerator '''
    data_df = pd.DataFrame({"image_names": image_names, "image_classes": image_classes})
    data_df.image_classes = data_df.image_classes.map(str)

    ''' ImageDataGenerator flow_from_dataframe '''
    batch_size = 128
    image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.05)
    train_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col="image_classes", class_mode='categorical', target_size=(112, 112), batch_size=batch_size, subset='training', validate_filenames=False)
    # Found 5240388 non-validated image filenames belonging to 85742 classes.
    val_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col="image_classes", class_mode='categorical', target_size=(112, 112), batch_size=batch_size, subset='validation', validate_filenames=False)
    # Found 582265 non-validated image filenames belonging to 85742 classes.

    classes = data_df.image_classes.unique().shape[0]
    steps_per_epoch = np.ceil(len(train_data_gen.classes) / batch_size)
    validation_steps = np.ceil(len(val_data_gen.classes) / batch_size)
    ```
    The training data is split into two datasets, `train_data_gen` and `val_data_gen`, we will just use `val_data_gen` for evaluation.
## Convert ImageDataGenerator to dataset
  - `tf.data.Dataset` can speed up data reading significantly
    ```py
    ''' Convert to tf.data.Dataset '''
    train_ds = tf.data.Dataset.from_generator(lambda: train_data_gen, output_types=(tf.float32, tf.int32), output_shapes=([None, 112, 112, 3], [None, classes]))
    # train_ds = train_ds.cache()
    train_ds = train_ds.repeat()
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(lambda: val_data_gen, output_types=(tf.float32, tf.int32), output_shapes=([None, 112, 112, 3], [None, classes]))

    xx, yy = next(iter(train_ds))
    print(xx.shape, yy.shape)
    # (128, 112, 112, 3) (128, 85742)
    ```
## Basic model
  - Our basic model will include:
    - A backbone like `ResNet50` or `ResNet101` or `MobileNet` or any others
    - A `GlobalAveragePooling2D` or `Flatten` layer
    - An `Dense` layer outputs `embedding` values will be used for classification
    - In this basic implementation, we will add different `bottleneck` for different loss functions
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
    basic_model.summary()
    ```
## Softmax loss train
  - This could be the simplest implementation for this face classification mission
    - Softmax train could set a baseline for our model, and the convergence speed is faster than other losses, so we could use this `basic model` for other loss training
    - Just notice if the `loss` is decreasing and `accuracy` is increasing, if this basic training is not working this way, then there must be something wrong
  - **Python code** we only add a `softmax` layer on top of the `basic model`
    ```py
    ''' Model definition '''
    output = layers.Dense(classes, activation='softmax')(basic_model.outputs[0])
    model = keras.models.Model(basic_model.inputs[0], output)
    model.summary()

    ''' Compile and fit '''
    model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=["accuracy"])
    hist = model.fit(train_ds, epochs=20, verbose=1, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps)
    ```
  - **Training process** just run a few step
    ```py
    Epoch 1/200
    43216/43216 [==============================] - 8650s 200ms/step - loss: 3.8481 - accuracy: 0.4496 - val_loss: 2.6660 - val_accuracy: 0.5180
    Epoch 2/200
    43216/43216 [==============================] - 8792s 203ms/step - loss: 0.9634 - accuracy: 0.8118 - val_loss: 1.2425 - val_accuracy: 0.7599
    Epoch 3/200
    43216/43216 [==============================] - 8720s 202ms/step - loss: 0.6660 - accuracy: 0.8676 - val_loss: 1.3942 - val_accuracy: 0.7380
    Epoch 4/200
    43216/43216 [==============================] - 8713s 202ms/step - loss: 0.5394 - accuracy: 0.8920 - val_loss: 0.6720 - val_accuracy: 0.8733
    Epoch 5/200
    43216/43216 [==============================] - 8873s 205ms/step - loss: 0.4662 - accuracy: 0.9063 - val_loss: 0.7837 - val_accuracy: 0.8540
    ```
## Arcface loss train
  - We have some realization only differ in detail, and there is no big difference in result
  - **Model definition** we add a `WeightNormalization` layer on top of `basic model`
    - The first realization is only normalize `weight` in this layer, and normalize `embedding` in loss function
    - The second realization is normalize both `weight` and `embedding` in this layer, we will use this in our basic implementation
    - In custom keras layer inheriting `keras.layers.Layer`, other function like `get_config` should also be implemented for model serialization
    ```py
    ''' WeightNormalization layer <second realization> '''
    class NormDense(tf.keras.layers.Layer):
        def __init__(self, classes=1000, **kwargs):
            super(NormDense, self).__init__(**kwargs)
            self.output_dim = classes
        def build(self, input_shape):
            self.w = self.add_weight(name='norm_dense_w', shape=(input_shape[-1], self.output_dim), initializer='random_normal', trainable=True)
            super(NormDense, self).build(input_shape)
        def call(self, inputs, **kwargs):
            norm_w = tf.nn.l2_normalize(self.w, axis=0)
            inputs = tf.nn.l2_normalize(inputs, axis=1) # comment this to turn off embedding normalization
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

    ''' Model definition '''
    inputs = basic_model.inputs[0]
    embedding = basic_model.outputs[0]
    output = NormDense(classes, name='norm_dense')(embedding)
    model = keras.models.Model(inputs, output)
    model.summary()
    ```
  - **Arcface loss function**
    - The first realization is like the one in original mxnet insightface
    - The second one is from other implementation of arcface
    ```py
    ''' Original mxnet insightface loss '''
    def arcface_loss(y_true, norm_logits, margin1=0.9, margin2=0.4, margin3=0.15, scale=64.0):
        theta = tf.acos(norm_logits)
        cond = tf.where(tf.greater(theta * margin1 + margin3, np.pi), tf.zeros_like(y_true), y_true)
        cond = tf.cast(cond, dtype=tf.bool)
        m1_theta_plus_m3 = tf.where(cond, theta * margin1 + margin3, theta)
        cos_m1_theta_plus_m3 = tf.cos(m1_theta_plus_m3)
        arcface_logits = tf.where(cond, cos_m1_theta_plus_m3 - margin2, cos_m1_theta_plus_m3) * scale
        # tf.assert_equal(tf.math.is_nan(tf.reduce_mean(arcface_logits)), False)
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=True)
    ```
    ```py
    ''' Another realization '''
    def arcface_loss(labels, norm_logits, s=64.0, m=0.45):
        cos_m = tf.math.cos(m)
        sin_m = tf.math.sin(m)
        mm = sin_m * m
        threshold = tf.math.cos(np.pi - m)

        cos_t2 = tf.square(norm_logits)
        sin_t2 = tf.subtract(1., cos_t2)
        sin_t = tf.sqrt(sin_t2)
        cos_mt = s * tf.subtract(tf.multiply(norm_logits, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
        cond_v = norm_logits - threshold
        cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
        keep_val = s * (norm_logits - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.cast(labels, tf.float32)
        inv_mask = tf.subtract(1., mask)
        s_cos_t = tf.multiply(s, norm_logits)
        arcface_logits = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask))
        return tf.keras.losses.categorical_crossentropy(labels, arcface_logits, from_logits=True)
    ```
  - **Model compile and fit**
    ```py
    model.compile(optimizer='adamax', loss=arcface_loss, metrics=["accuracy"])
    hist = model.fit(train_ds, epochs=20, verbose=1, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps)
    ```
  - **Training process** just run a few step. We can see the loss is decreasing, but the accuracy is increasing too slow
    ```py
    Epoch 1/200
    43216/43216 [==============================] - 9148s 212ms/step - loss: 34.7171 - accuracy: 0.0088 - val_loss: 14.4753 - val_accuracy: 0.0010
    Epoch 2/200
    43216/43216 [==============================] - 8995s 208ms/step - loss: 14.4870 - accuracy: 0.0022 - val_loss: 14.2573 - val_accuracy: 0.0118
    Epoch 3/200
    43216/43216 [==============================] - 8966s 207ms/step - loss: 14.5741 - accuracy: 0.0146 - val_loss: 14.6334 - val_accuracy: 0.0156
    Epoch 4/200
    43216/43216 [==============================] - 8939s 207ms/step - loss: 14.6519 - accuracy: 0.0175 - val_loss: 14.3232 - val_accuracy: 0.0158
    Epoch 5/200
    43216/43216 [==============================] - 9122s 211ms/step - loss: 14.6973 - accuracy: 0.0198 - val_loss: 15.0081 - val_accuracy: 0.0210
    ```
## Combined train
  - We use the same basic model for `softmax` and `arcface` train, so we can use a trained `basic model` by `softmax`, and fine-tune by `acrface` loss
    ```py
    ''' Save basic model weights in softmax loss train '''
    basic_model.save_weights('basi_model_softmax.h5')

    ''' Load basic model weights before arcface train fit '''
    basic_model.load_weights('basi_model_softmax.h5')

    ''' Train for a few epoch only to fit the bottleneck layer '''
    basic_model.trainable = False
    model.compile(optimizer='adamax', loss=arcface_loss, metrics=["accuracy"])
    hist = model.fit(train_ds, epochs=2, verbose=1, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps)

    ''' Train the whole model '''
    basic_model.trainable = True
    model.compile(optimizer='adamax', loss=arcface_loss, metrics=["accuracy"])  # MUST run compile after changing trainable value
    hist = model.fit(train_ds, epochs=20, verbose=1, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps)
    ```
***
