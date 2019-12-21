- [TensorFlow Addons Losses: TripletSemiHardLoss](https://www.tensorflow.org/addons/tutorials/losses_triplet)
- [TensorFlow Addons Layers: WeightNormalization](https://www.tensorflow.org/addons/tutorials/layers_weightnormalization)

# Keras Insightface
## MXnet record to folder
  ```py
  import os
  import numpy as np
  import mxnet as mx
  from tqdm import tqdm

  # read_dir = '/datasets/faces_glint/'
  # save_dir = '/datasets/faces_glint_112x112_folders'
  read_dir = '/datasets/faces_emore'
  save_dir = '/datasets/faces_emore_112x112_folders'
  idx_path = os.path.join(read_dir, 'train.idx')
  bin_path = os.path.join(read_dir, 'train.rec')

  imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
  rec_header, _ = mx.recordio.unpack(imgrec.read_idx(0))

  # for ii in tqdm(range(1, 10)):
  for ii in tqdm(range(1, int(rec_header.label[0]))):
      img_info = imgrec.read_idx(ii)
      header, img = mx.recordio.unpack(img_info)
      img_idx = str(int(np.sum(header.label)))
      img_save_dir = os.path.join(save_dir, img_idx)
      if not os.path.exists(img_save_dir):
          os.makedirs(img_save_dir)
      # print(os.path.join(img_save_dir, str(ii) + '.jpg'))
      with open(os.path.join(img_save_dir, str(ii) + '.jpg'), 'wb') as ff:
          ff.write(img)
  ```
  ```py
  import io
  import pickle
  import tensorflow as tf
  from skimage.io import imread

  test_bin_file = '/datasets/faces_emore/agedb_30.bin'
  test_bin_file = '/datasets/faces_emore/cfp_fp.bin'
  with open(test_bin_file, 'rb') as ff:
      bins, issame_list = pickle.load(ff, encoding='bytes')

  bb = [tf.image.encode_jpeg(imread(io.BytesIO(ii))) for ii in bins]
  with open(test_bin_file, 'wb') as ff:
      pickle.dump([bb, issame_list], ff)
  ```
## Loading data by ImageDataGenerator
  ```py
  ''' flow_from_dataframe '''
  import glob2
  import pickle
  image_names = glob2.glob('/datasets/faces_emore_112x112_folders/*/*.jpg')
  image_names = np.random.permutation(image_names).tolist()
  image_classes = [int(os.path.basename(os.path.dirname(ii))) for ii in image_names]

  with open('faces_emore_img_class_shuffle.pkl', 'wb') as ff:
      pickle.dump({'image_names': image_names, "image_classes": image_classes}, ff)

  import pickle
  from keras.preprocessing.image import ImageDataGenerator
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  with open('faces_emore_img_class_shuffle.pkl', 'rb') as ff:
      aa = pickle.load(ff)
  image_names, image_classes = aa['image_names'], aa['image_classes']
  image_names = np.random.permutation(image_names).tolist()
  image_classes = [int(os.path.basename(os.path.dirname(ii))) for ii in image_names]
  print(len(image_names), len(image_classes))
  # 5822653 5822653

  data_df = pd.DataFrame({"image_names": image_names, "image_classes": image_classes})
  data_df.image_classes = data_df.image_classes.map(str)
  # image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.1)
  image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.05)
  train_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col="image_classes", class_mode='categorical', target_size=(112, 112), batch_size=128, subset='training', validate_filenames=False)
  # Found 5240388 non-validated image filenames belonging to 85742 classes.
  val_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col="image_classes", class_mode='categorical', target_size=(112, 112), batch_size=128, subset='validation', validate_filenames=False)
  # Found 582265 non-validated image filenames belonging to 85742 classes.

  classes = data_df.image_classes.unique().shape[0]
  steps_per_epoch = np.ceil(len(train_data_gen.classes) / 128)
  validation_steps = np.ceil(len(val_data_gen.classes) / 128)

  ''' Convert to tf.data.Dataset '''
  train_ds = tf.data.Dataset.from_generator(lambda: train_data_gen, output_types=(tf.float32, tf.int32), output_shapes=([None, 112, 112, 3], [None, classes]))
  # train_ds = train_ds.cache()
  # train_ds = train_ds.shuffle(buffer_size=128 * 1000)
  train_ds = train_ds.repeat()
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

  val_ds = tf.data.Dataset.from_generator(lambda: val_data_gen, output_types=(tf.float32, tf.int32), output_shapes=([None, 112, 112, 3], [None, classes]))

  xx, yy = next(iter(train_ds))
  print(xx.shape, yy.shape)
  # (128, 112, 112, 3) (128, 85742)
  ```
## Loading data by Datasets
  ```py
  import glob2
  import pickle
  image_names = glob2.glob('/datasets/faces_emore_112x112_folders/*/*.jpg')
  image_names = np.random.permutation(image_names).tolist()
  image_classes = [int(os.path.basename(os.path.dirname(ii))) for ii in image_names]

  with open('faces_emore_img_class_shuffle.pkl', 'wb') as ff:
      pickle.dump({'image_names': image_names, "image_classes": image_classes}, ff)

  import pickle
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  with open('faces_emore_img_class_shuffle.pkl', 'rb') as ff:
      aa = pickle.load(ff)
  image_names, image_classes = aa['image_names'], aa['image_classes']
  classes = np.max(image_classes) + 1
  print(len(image_names), len(image_classes), classes)
  # 5822653 5822653 85742

  # list_ds = tf.data.Dataset.list_files('/datasets/faces_emore_112x112_folders/*/*')
  list_ds = tf.data.Dataset.from_tensor_slices(image_names)

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
## Evaluate
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
  ```py
  class mi_basic_model:
      def __init__(self):
          self.predict = lambda xx: interf(xx)['embedding'].numpy()
      def save(self, path):
          print('Saved to %s' % (path))
  basic_model = mi_basic_model()

  aa = epoch_eval_callback('/datasets/faces_emore/agedb_30.bin', save_model='./test')
  aa = epoch_eval_callback('/home/leondgarse/workspace/datasets/faces_emore/lfw.bin')
  aa.on_epoch_end()
  ```
  ```py
  # basic_model_centsoft_0_split.h5
  >>>> lfw evaluation max accuracy: 0.992833, thresh: 0.188595, overall max accuracy: 0.992833
  >>>> cfp_fp evaluation max accuracy: 0.909571, thresh: 0.119605, overall max accuracy: 0.909571
  >>>> agedb_30 evaluation max accuracy: 0.887500, thresh: 0.238278, overall max accuracy: 0.887500
  ```
  ```py
  # basic_model_arc_8_split.h5
  >>>> lfw evaluation max accuracy: 0.994167, thresh: 0.141986, overall max accuracy: 0.994167
  >>>> cfp_fp evaluation max accuracy: 0.867429, thresh: 0.106673, overall max accuracy: 0.867429
  >>>> agedb_30 evaluation max accuracy: 0.902167, thresh: 0.128596, overall max accuracy: 0.902167
  ```
  ```py
  # basic_model_arc_split.h5
  >>>> lfw evaluation max accuracy: 0.993000, thresh: 0.125761, overall max accuracy: 0.993000
  >>>> agedb_30 evaluation max accuracy: 0.912667, thresh: 0.084312, overall max accuracy: 0.912667
  >>>> cfp_fp evaluation max accuracy: 0.859143, thresh: 0.068290, overall max accuracy: 0.859143
  ```
## Basic model
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

  ''' Callbacks '''
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

  # lfw_eval = epoch_eval_callback('/datasets/faces_emore/lfw.bin')
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

  ''' Loss function wrapper '''
  def logits_accuracy(y_true, y_pred):
      logits = y_pred[:, 512:]
      return keras.metrics.categorical_accuracy(y_true, logits)
  ```
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
## Softmax train
  ```py
  def softmax_loss(y_true, y_pred):
      logits = y_pred[:, 512:]
      return keras.losses.categorical_crossentropy(y_true, logits, from_logits=True)

  with mirrored_strategy.scope():
      model.compile(optimizer='adamax', loss=softmax_loss, metrics=[logits_accuracy])
  ```
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
## Arcface loss
  ```py
  # def arcface_loss(y_true, y_pred, margin1=1.0, margin2=0.2, margin3=0.3, scale=64.0):
  def arcface_loss(y_true, y_pred, margin1=0.9, margin2=0.4, margin3=0.15, scale=64.0):
      # y_true = tf.squeeze(y_true)
      # y_true = tf.cast(y_true, tf.int32)
      # y_true = tf.argmax(y_true, 1)
      # cos_theta = tf.nn.l2_normalize(logits, axis=1)
      # theta = tf.acos(cos_theta)
      # mask = tf.one_hot(y_true, epth=norm_logits.shape[-1])
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
      tf.assert_equal(tf.math.is_nan(tf.reduce_mean(arcface_logits)), False)
      return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=True)

  with mirrored_strategy.scope():
      model.compile(optimizer='adamax', loss=arcface_loss, metrics=[logits_accuracy])
  ```
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
## Arcface loss 2
  ```py
  def arcface_loss(labels, norm_logits, s=64.0, m=0.45):
      # labels = tf.squeeze(labels)
      # labels = tf.cast(labels, tf.int32)
      # norm_logits = tf.nn.l2_normalize(logits, axis=1)

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
      # mask = tf.one_hot(labels, depth=norm_logits.shape[-1])
      mask = tf.cast(labels, tf.float32)
      inv_mask = tf.subtract(1., mask)
      s_cos_t = tf.multiply(s, norm_logits)
      arcface_logits = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask))
      # return tf.keras.losses.sparse_categorical_crossentropy(labels, arcface_logits, from_logits=True)
      return tf.keras.losses.categorical_crossentropy(labels, arcface_logits, from_logits=True)
  ```
## Center loss
  ```py
  class Save_Numpy_Callback(tf.keras.callbacks.Callback):
      def __init__(self, save_file, save_tensor):
          super(Save_Numpy_Callback, self).__init__()
          self.save_file = os.path.splitext(save_file)[0]
          self.save_tensor = save_tensor

      def on_epoch_end(self, epoch=0, logs=None):
          np.save(self.save_file, self.save_tensor.numpy())

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
          # print(centers_batch.shape, self.centers.shape, labels.shape, diff.shape)
          self.centers.assign(tf.tensor_scatter_nd_sub(self.centers, tf.expand_dims(labels, 1), diff))
          # centers_batch = tf.gather(self.centers, labels)

          return loss * self.factor

  center_loss = Center_loss(classes, factor=1.0, initial_file='./centers.npy')
  callbacks.append(center_loss.save_centers_callback)

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

  with mirrored_strategy.scope():
      # model.compile(optimizer='adamax', loss=single_center_loss, metrics=[logits_accuracy()])
      model.compile(optimizer='adamax', loss=center_softmax_loss, metrics=[logits_accuracy()])
  ```
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
## Offline Triplet loss train SUB
  ```py
  import pickle
  with open('faces_emore_img_class_shuffle.pkl', 'rb') as ff:
      aa = pickle.load(ff)
  image_names, image_classes = aa['image_names'], aa['image_classes']
  classes = np.max(image_classes) + 1
  print(len(image_names), len(image_classes), classes)
  # 5822653 5822653 85742

  from sklearn.preprocessing import normalize
  from tqdm import tqdm
  import pandas as pd

  class Triplet_datasets:
      def __init__(self, image_names, image_classes, batch_size=128, alpha=0.2, image_per_class=4, max_class=10000):
          self.AUTOTUNE = tf.data.experimental.AUTOTUNE
          self.image_dataframe = pd.DataFrame({'image_names': image_names, "image_classes" : image_classes})
          self.classes = self.image_dataframe.image_classes.unique().shape[0]
          self.image_per_class = image_per_class
          self.max_class = max_class
          self.alpha = alpha
          self.batch_size = batch_size
          self.sub_total = np.ceil(self.max_class * image_per_class / batch_size)
          # self.update_triplet_datasets()

      def update_triplet_datasets(self):
          list_ds = self.prepare_sub_list_dataset()
          anchors, poses, negs = self.mine_triplet_data_pairs(list_ds)
          # self.train_dataset, self.steps_per_epoch = self.gen_triplet_train_dataset(anchors, poses, negs)
          return self.gen_triplet_train_dataset(anchors, poses, negs)

      def image_pick_func(self, df):
          vv = df.image_names.values
          choice_replace = vv.shape[0] < self.image_per_class
          return np.random.choice(vv, self.image_per_class, replace=choice_replace)

      def process_path(self, img_name, img_shape=(112, 112)):
          parts = tf.strings.split(img_name, os.path.sep)[-2]
          label = tf.cast(tf.strings.to_number(parts), tf.int32)
          img = tf.io.read_file(img_name)
          img = tf.image.decode_jpeg(img, channels=3)
          img = tf.image.convert_image_dtype(img, tf.float32)
          img = tf.image.resize(img, img_shape)
          img = tf.image.random_flip_left_right(img)
          return img, label, img_name

      def prepare_sub_list_dataset(self):
          tt = self.image_dataframe.groupby("image_classes").apply(self.image_pick_func)
          sub_tt = tt[np.random.choice(tt.shape[0], self.max_class, replace=False)]
          cc = np.concatenate(sub_tt.values)
          list_ds = tf.data.Dataset.from_tensor_slices(cc)
          list_ds = list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
          list_ds = list_ds.batch(self.batch_size)
          list_ds = list_ds.prefetch(buffer_size=self.AUTOTUNE)
          return list_ds

      def batch_triplet_image_process(self, anchors, poses, negs):
          anchor_labels = tf.zeros_like(anchors, dtype=tf.float32)
          labels = tf.concat([anchor_labels, anchor_labels + 1, anchor_labels + 2], 0)
          image_names = tf.concat([anchors, poses, negs], 0)
          images = tf.map_fn(lambda xx: self.process_path(xx)[0], image_names, dtype=tf.float32)
          # image_classes = tf.map_fn(lambda xx: tf.strings.split(xx, os.path.sep)[-2], image_names)
          # return images, labels, image_classes
          return images, labels

      def mine_triplet_data_pairs(self, list_ds):
          embs, labels, img_names = [], [], []
          for imgs, label, img_name in tqdm(list_ds, "Embedding", total=self.sub_total):
              emb = basic_model.predict(imgs)
              embs.extend(emb)
              labels.extend(label.numpy())
              img_names.extend(img_name.numpy())
          embs = np.array(embs)
          not_nan_choice = np.isnan(embs).sum(1) == 0
          embs = embs[not_nan_choice]
          embs = normalize(embs)
          labels = np.array(labels)[not_nan_choice]
          img_names = np.array(img_names)[not_nan_choice]

          '''
          where we have same label: pos_idx --> [10, 11, 12, 13]
          image names: pose_imgs --> ['a', 'b', 'c', 'd']
          anchor <--> pos: {10: [11, 12, 13], 11: [12, 13], 12: [13]}
          distance of anchor and pos: stack_pos_dists -->
              [[10, 11], [10, 12], [10, 13], [11, 12], [11, 13], [12, 13]]
          anchors image names: stack_anchor_name --> ['a', 'a', 'a', 'b', 'b', 'c']
          pos image names: stack_pos_name --> ['b', 'c', 'd', 'c', 'd', 'd']
          distance between anchor and all others: stack_dists -->
              [d(10), d(10), d(10), d(11), d(11), d(12)]
          distance between pos and neg for all anchor: neg_pos_dists -->
              [d([10, 11]) - d(10), d([10, 12]) - d(10), d([10, 13]) - d(10),
               d([11, 12]) - d(11), d([11, 13]) - d(11),
               d([12, 13]) - d(12)]
          valid pos indexes: neg_valid_x --> [0, 0, 0, 1, 1, 1, 2, 5, 5, 5]
          valid neg indexss: neg_valid_y --> [1022, 312, 3452, 6184, 294, 18562, 82175, 9945, 755, 8546]
          unique valid pos indexes: valid_pos --> [0, 1, 2, 5]
          random valid neg indexs in each pos: valid_neg --> [1022, 294, 82175, 8546]
          anchor names: stack_anchor_name[valid_pos] --> ['a', 'a', 'a', 'c']
          pos names: stack_pos_name[valid_pos] --> ['b', 'c', 'd', 'd']
          '''
          anchors, poses, negs = [], [], []
          for label in tqdm(np.unique(labels), "Mining triplet pairs"):
          # for label in np.unique(labels):
              pos_idx = np.where(labels == label)[0]
              pos_imgs = img_names[pos_idx]
              total = pos_idx.shape[0]
              pos_embs = embs[pos_idx[:-1]]
              dists = np.dot(pos_embs, embs.T)
              pos_dists = [dists[id, pos_idx[id + 1:]] for id in range(total - 1)]
              stack_pos_dists = np.expand_dims(np.hstack(pos_dists), -1)

              elem_repeats = np.arange(1, total)[::-1]
              stack_anchor_name = pos_imgs[:-1].repeat(elem_repeats, 0)
              stack_pos_name = np.hstack([pos_imgs[ii:] for ii in range(1, total)])
              stack_dists = dists.repeat(elem_repeats, 0)

              neg_pos_dists = stack_pos_dists - stack_dists - self.alpha
              neg_pos_dists[:, pos_idx] = 1
              neg_valid_x, neg_valid_y = np.where(neg_pos_dists < 0)

              if len(neg_valid_x) > 0:
                  valid_pos = np.unique(neg_valid_x)
                  valid_neg = [np.random.choice(neg_valid_y[neg_valid_x == ii]) for ii in valid_pos]
                  anchors.extend(stack_anchor_name[valid_pos])
                  poses.extend(stack_pos_name[valid_pos])
                  negs.extend(img_names[valid_neg])
                  # self.minning_print_func(pos_imgs, valid_pos, valid_neg, stack_anchor_name, stack_pos_name, labels, stack_dists)
          print(">>>> %d triplets found." % (len(anchors)))
          return anchors, poses, negs

      def gen_triplet_train_dataset(self, anchors, poses, negs):
          num_triplets = len(anchors)
          train_dataset = tf.data.Dataset.from_tensor_slices((anchors, poses, negs))
          train_dataset = train_dataset.shuffle(num_triplets + 1)
          train_dataset = train_dataset.batch(self.batch_size)
          train_dataset = train_dataset.map(self.batch_triplet_image_process, num_parallel_calls=self.AUTOTUNE)
          train_dataset = train_dataset.prefetch(buffer_size=self.AUTOTUNE)
          steps_per_epoch = np.ceil(num_triplets / self.batch_size)
          return train_dataset, steps_per_epoch

      def minning_print_func(self, pose_imgs, valid_pos, valid_neg, stack_anchor_name, stack_pos_name, labels, stack_dists):
          img2idx = dict(zip(pose_imgs, range(len(pose_imgs))))
          valid_anchor_idx = [img2idx[stack_anchor_name[ii]] for ii in valid_pos]
          valid_pos_idx = [img2idx[stack_pos_name[ii]] for ii in valid_pos]
          print("anchor: %s" % (list(zip(valid_anchor_idx, labels[pos_idx[valid_anchor_idx]]))))
          print("pos: %s" % (list(zip(valid_pos_idx, labels[pos_idx[valid_pos_idx]]))))
          print("neg: %s" % (labels[valid_neg]))
          print("pos dists: %s" % ([stack_dists[ii, pos_idx[jj]] for ii, jj in zip(valid_pos, valid_pos_idx)]))
          print("neg dists: %s" % ([stack_dists[ii, jj] for ii, jj in zip(valid_pos, valid_neg)]))
          print()

  def triplet_loss(labels, embeddings, alpha=0.2):
      labels = tf.squeeze(labels)
      labels.set_shape([None])
      anchor_emb = tf.nn.l2_normalize(embeddings[labels == 0], 1)
      pos_emb = tf.nn.l2_normalize(embeddings[labels == 1], 1)
      neg_emb = tf.nn.l2_normalize(embeddings[labels == 2], 1)
      pos_dist = tf.reduce_sum(tf.multiply(anchor_emb, pos_emb), -1)
      neg_dist = tf.reduce_sum(tf.multiply(anchor_emb, neg_emb), -1)
      basic_loss = neg_dist - pos_dist + alpha
      return tf.reduce_mean(tf.maximum(basic_loss, 0.0), axis=0)

  basic_model.compile(optimizer='adamax', loss=triplet_loss)
  triplet_datasets = Triplet_datasets(image_names, image_classes, image_per_class=5, max_class=10000)
  for epoch in range(100):
      train_dataset, steps_per_epoch = triplet_datasets.update_triplet_datasets()
      basic_model.fit(train_dataset, epochs=1, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, initial_epoch=epoch, use_multiprocessing=True, workers=4)
  ```
  ```py
  def mine_triplet_data_pairs(embs, labels, img_names, alpha=0.2):
      anchors, poses, negs = [], [], []
      for idx, (emb, label) in enumerate(zip(embs, labels)):
          dist = np.dot(emb, embs.T)
          pos_indexes = np.where(labels == label)[0]
          pos_indexes = pos_indexes[pos_indexes > idx]
          neg_indxes = np.where(labels != label)[0]
          for pos in pos_indexes:
              if pos == idx:
                  continue
              pos_dist = dist[pos]
              neg_valid = neg_indxes[pos_dist - dist[neg_indxes] < alpha]
              if neg_valid.shape[0] == 0:
                  continue
              neg_random = np.random.choice(neg_valid)
              anchors.append(img_names[idx])
              poses.append(img_names[pos])
              negs.append(img_names[neg_random])
              print("label: %d, pos: %d, %f, neg: %d, %f" % (label, labels[pos], dist[pos], labels[neg_random], dist[neg_random]))
      return anchors, poses, negs
  ```
## tf-insightface train
  - Arcface loss
    ```py
    epoch: 25, step: 60651, loss = 15.109766960144043, logit_loss = 15.109766960144043, center_loss = 0
    epoch: 25, step: 60652, loss = 17.565662384033203, logit_loss = 17.565662384033203, center_loss = 0
    Saving checkpoint for epoch 25 at /home/tdtest/workspace/tf_insightface/recognition/mymodel-26
    Time taken for epoch 25 is 8373.555536031723 sec
    ```
  - Arcface center loss
    ```py
    epoch: 0, step: 60652, loss = 11.264640808105469, logit_loss = 11.262413024902344, center_loss = 0.002227420685812831
    Saving checkpoint for epoch 0 at /home/tdtest/workspace/tf_insightface/recognition/mymodel-1
    Time taken for epoch 0 is 8373.187638521194 sec
    ```
## FUNC
  - **tf.compat.v1.scatter_sub** 将 `ref` 中 `indices` 指定位置的值减去 `updates`，会同步更新 `ref`
    ```py
    scatter_sub(ref, indices, updates, use_locking=False, name=None)
    ```
    ```py
    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8],dtype = tf.int32)
    indices = tf.constant([4, 3, 1, 7],dtype = tf.int32)
    updates = tf.constant([9, 10, 11, 12],dtype = tf.int32)
    print(tf.compat.v1.scatter_sub(ref, indices, updates).numpy())
    # [ 1 -9  3 -6 -4  6  7 -4]
    print(ref.numpy())
    [ 1 -9  3 -6 -4  6  7 -4]
    ```
  - **tf.tensor_scatter_nd_sub** 多维数据的 `tf.compat.v1.scatter_sub`
    ```py
    tensor = tf.ones([8], dtype=tf.int32)
    indices = tf.constant([[4], [3], [1] ,[7]])
    updates = tf.constant([9, 10, 11, 12])
    print(tf.tensor_scatter_nd_sub(tensor, indices, updates).numpy())
    # [ 1 -9  3 -6 -4  6  7 -4]
    ```
  - **tf.gather** 根据 `indices` 切片选取 `params` 中的值
    ```py
    gather_v2(params, indices, validate_indices=None, axis=None, batch_dims=0, name=None)
    ```
    ```py
    print(tf.gather([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [0, 1, 0]).numpy())
    # [[1 2 3] [4 5 6] [1 2 3]]
    ```
  - **l2 normalize**
    ```py
    aa = [x1, x2]
    bb = [[y1, y2], [y3, y4]]

    ''' tf.nn.l2_normalize(tf.matmul(aa, bb)) '''
    tf.matmul(aa, bb) = [x1 * y1 + x2 * y3, x1 * y2 + x2 * y4]
    tf.nn.l2_normalize(tf.matmul(aa, bb)) = [
        (x1 * y1 + x2 * y3) / sqrt((x1 * y1 + x2 * y3) ** 2 + (x1 * y2 + x2 * y4) ** 2)
        (x1 * y2 + x2 * y4) / sqrt((x1 * y1 + x2 * y3) ** 2 + (x1 * y2 + x2 * y4) ** 2)
    ]

    ''' tf.matmul(tf.nn.l2_normalize(aa), tf.nn.l2_normalize(bb)) '''
    tf.nn.l2_normalize(aa) = [x1 / sqrt(x1 ** 2 + x2 ** 2), x2 / sqrt(x1 ** 2 + x2 ** 2)]
    tf.nn.l2_normalize(bb) = [[y1 / sqrt(y1 ** 2 + y3 ** 2), y2 / sqrt(y2 ** 2 + y4 ** 2)],
                              [y3 / sqrt(y1 ** 2 + y3 ** 2), y4 / sqrt(y2 ** 2 + y4 ** 2)]]
    tf.matmul(tf.nn.l2_normalize(aa), tf.nn.l2_normalize(bb)) = [
        (x1 * y1 + x2 * y3) / sqrt((x1 ** 2 + x2 ** 2) * (y1 ** 2 + y3 ** 2)),
        (x1 * y2 + x2 * y4) / sqrt((x1 ** 2 + x2 ** 2) * (y2 ** 2 + y4 ** 2))
    ]
    ```
    ```py
    aa = tf.convert_to_tensor([[1, 2]], dtype='float32')
    bb = tf.convert_to_tensor(np.arange(4).reshape(2, 2), dtype='float32')
    print(aa.numpy())
    # [[1. 2.]]
    print(bb.numpy())
    # [[0. 1.] [2. 3.]]

    print(tf.matmul(aa, bb).numpy())
    # [[4. 7.]]
    print(tf.nn.l2_normalize(tf.matmul(aa, bb), axis=1).numpy())
    # [[0.49613893 0.8682431 ]]

    print(tf.nn.l2_normalize(aa, 1).numpy())
    # [[0.4472136 0.8944272]]
    print(tf.nn.l2_normalize(bb, 0).numpy())
    # [[0.         0.31622776] [1.         0.94868326]]
    print(tf.matmul(tf.nn.l2_normalize(aa, 1), tf.nn.l2_normalize(bb, 0)).numpy())
    # [[0.8944272  0.98994946]]
    ```
***
