## Basic model
  ```py
  import multiprocessing as mp
  mp.set_start_method('forkserver')
  hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, initial_epoch=11, use_multiprocessing=True, workers=4)

  # hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch)
  hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps)

  hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps)
  ```
  ```py
  from tensorflow.keras import layers
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

  ''' Basic model '''
  # Multi GPU
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
      # xx = keras.applications.ResNet101V2(include_top=False, weights='imagenet')
      # xx = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
      xx = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
      xx.trainable = True

      inputs = xx.inputs[0]
      nn = xx.outputs[0]
      nn = layers.GlobalAveragePooling2D()(nn)
      nn = layers.Dropout(0.1)(nn)
      embedding = layers.Dense(512, name='embedding')(nn)
      basic_model = keras.models.Model(inputs, embedding)

      basic_model.load_weights('basic_model_arc_8_split.h5')
      basic_model.trainable = False
  ```
