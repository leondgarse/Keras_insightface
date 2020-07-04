import data
import data_gen
import evals
import losses
import myCallbacks
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import os

import multiprocessing as mp

mp.set_start_method("forkserver")

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

def print_buildin_models():
    print("""
    >>>> buildin_models
    mobilenet, mobilenetv2, mobilenetv3_small, mobilenetv3_large, mobilefacenet, se_mobilefacenet, nasnetmobile
    resnet50v2, resnet101v2, se_resnext, resnest50, resnest101,
    efficientnetb0, efficientnetb1, efficientnetb2, efficientnetb3, efficientnetb4, efficientnetb5, efficientnetb6, efficientnetb7,
    """, end='')


def buildin_models(name, dropout=1, emb_shape=512, input_shape=(112, 112, 3), **kwargs):
    name = name.lower()
    """ Basic model """
    if name == "mobilenet":
        xx = keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name == "mobilenetv2":
        xx = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name == "resnet50v2":
        xx = keras.applications.ResNet50V2(input_shape=input_shape, include_top=False, weights="imagenet", **kwargs)
    elif name == "resnet101v2":
        xx = keras.applications.ResNet101V2(input_shape=input_shape, include_top=False, weights="imagenet", **kwargs)
    elif name == "nasnetmobile":
        xx = keras.applications.NASNetMobile(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name.startswith("efficientnet"):
        if "-dev" in tf.__version__:
            import tensorflow.keras.applications.efficientnet as efntf
        else:
            import efficientnet.tfkeras as efntf

        if name[-2] == "b":
            compound_scale = int(name[-1])
            models = [
                efntf.EfficientNetB0,
                efntf.EfficientNetB1,
                efntf.EfficientNetB2,
                efntf.EfficientNetB3,
                efntf.EfficientNetB4,
                efntf.EfficientNetB5,
                efntf.EfficientNetB6,
                efntf.EfficientNetB7,
            ]
            model = models[compound_scale]
        else:
            model = efntf.EfficientNetL2
        xx = model(weights="imagenet", include_top=False, input_shape=input_shape)  # or weights='noisy-student'
    elif name.startswith("se_resnext"):
        from keras_squeeze_excite_network import se_resnext

        if name.endswith("101"):  # se_resnext101
            depth = [3, 4, 23, 3]
        else:  # se_resnext50
            depth = [3, 4, 6, 3]
        xx = se_resnext.SEResNextImageNet(weights="imagenet", input_shape=input_shape, include_top=False, depth=depth)
    elif name.startswith("resnest"):
        from backbones import resnest

        if name == "resnest50":
            xx = resnest.ResNest50(input_shape=input_shape)
        else:
            xx = resnest.ResNest101(input_shape=input_shape)
    elif name.startswith("mobilenetv3"):
        from backbones import mobilenetv3

        size = 'small' if 'small' in name else 'large'
        xx = mobilenetv3.MobilenetV3(input_shape=input_shape, include_top=False, size=size)
    elif "mobilefacenet" in name or "mobile_facenet" in name:
        from backbones import mobile_facenet

        use_se = True if 'se' in name else False
        xx = mobile_facenet.mobile_facenet(input_shape=input_shape, include_top=False, name=name, use_se=use_se)
    else:
        return None
    # xx = keras.models.load_model('checkpoints/mobilnet_v1_basic_922667.h5', compile=False)
    xx.trainable = True

    inputs = xx.inputs[0]
    nn = xx.outputs[0]
    # nn = keras.layers.Conv2D(emb_shape, xx.output_shape[1], use_bias=False)(nn)

    """ GDC """
    nn = keras.layers.Conv2D(512, 1, use_bias=False)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.PReLU(shared_axes=[1, 2])(nn)
    nn = keras.layers.DepthwiseConv2D(int(nn.shape[1]), depth_multiplier=1, use_bias=False)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Conv2D(emb_shape, 1, use_bias=False, activation=None)(nn)
    if dropout > 0 and dropout < 1:
        nn = keras.layers.Dropout(dropout)(nn)
    nn = keras.layers.Flatten()(nn)
    # nn = keras.layers.Dense(emb_shape, activation=None, use_bias=False, kernel_initializer="glorot_normal")(nn)
    embedding = keras.layers.BatchNormalization(name="embedding")(nn)
    # norm_emb = layers.Lambda(tf.nn.l2_normalize, name='norm_embedding', arguments={'axis': 1})(embedding)
    basic_model = keras.models.Model(inputs, embedding, name=xx.name)
    return basic_model


class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, **kwargs):
        super(NormDense, self).__init__(**kwargs)
        self.init = keras.initializers.glorot_normal()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="norm_dense_w", shape=(input_shape[-1], self.units), initializer=self.init, trainable=True
        )
        super(NormDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        norm_w = K.l2_normalize(self.w, axis=0)
        inputs = K.l2_normalize(inputs, axis=1)
        return K.dot(inputs, norm_w)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(NormDense, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Train:
    def __init__(
        self,
        data_path,
        save_path,
        eval_paths=[],
        basic_model=None,
        model=None,
        compile=True,
        batch_size=128,
        lr_base=0.001,
        lr_decay=0.05,  # lr_decay < 1 for exponential, or it's cosine decay_steps
        lr_on_batch=0,  # lr_on_batch < 1 for update lr on epoch, or update on every [NUM] batches
        lr_min=0,
        eval_freq=1,
        random_status=0,
        custom_objects={},
    ):
        custom_objects.update(
            {
                "NormDense": NormDense,
                "margin_softmax": losses.margin_softmax,
                "arcface_loss": losses.arcface_loss,
                "ArcfaceLoss": losses.ArcfaceLoss,
                "CenterLoss": losses.CenterLoss,
                "batch_hard_triplet_loss": losses.batch_hard_triplet_loss,
                "batch_all_triplet_loss": losses.batch_all_triplet_loss,
                "BatchHardTripletLoss": losses.BatchHardTripletLoss,
                "BatchAllTripletLoss": losses.BatchAllTripletLoss,
                "logits_accuracy": self.logits_accuracy,
            }
        )
        self.model, self.basic_model, self.save_path = None, None, save_path
        if isinstance(model, str):
            if model.endswith(".h5") and os.path.exists(model):
                print(">>>> Load model from h5 file: %s..." % model)
                with keras.utils.custom_object_scope(custom_objects):
                    self.model = keras.models.load_model(model, compile=compile, custom_objects=custom_objects)
                basic_model = basic_model if basic_model is not None else self.__search_embedding_layer__(self.model)
                self.basic_model = keras.models.Model(self.model.inputs[0], self.model.layers[basic_model].output)
                self.model.summary()
        elif isinstance(model, keras.models.Model):
            self.model = model
            basic_model = basic_model if basic_model is not None else self.__search_embedding_layer__(self.model)
            self.basic_model = keras.models.Model(self.model.inputs[0], self.model.layers[basic_model].output)
        elif isinstance(basic_model, str):
            if basic_model.endswith(".h5") and os.path.exists(basic_model):
                print(">>>> Load basic_model from h5 file: %s..." % basic_model)
                with keras.utils.custom_object_scope(custom_objects):
                    self.basic_model = keras.models.load_model(basic_model, compile=compile, custom_objects=custom_objects)
        elif isinstance(basic_model, keras.models.Model):
            self.basic_model = basic_model

        if self.basic_model == None:
            print(
                "Initialize model by:\n"
                "| basic_model                                                     | model           |\n"
                "| --------------------------------------------------------------- | --------------- |\n"
                "| model structure                                                 | None            |\n"
                "| basic model .h5 file                                            | None            |\n"
                "| None for 'embedding' layer or layer index of basic model output | model .h5 file  |\n"
                "| None for 'embedding' layer or layer index of basic model output | model structure |\n"
            )
            return

        self.softmax, self.arcface, self.triplet = "softmax", "arcface", "triplet"

        self.batch_size = batch_size
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
            self.batch_size = batch_size * strategy.num_replicas_in_sync
            print(">>>> num_replicas_in_sync: %d, batch_size: %d" % (strategy.num_replicas_in_sync, self.batch_size))
        self.data_path, self.random_status = data_path, random_status
        self.train_ds, self.steps_per_epoch, self.classes = None, None, 0
        self.is_triplet_dataset = False
        self.default_optimizer = "adam"
        self.metrics = ["accuracy"]
        my_evals = [evals.eval_callback(self.basic_model, ii, batch_size=batch_size, eval_freq=eval_freq) for ii in eval_paths]
        if len(my_evals) != 0:
            my_evals[-1].save_model = os.path.splitext(save_path)[0]
        basic_callbacks = myCallbacks.basic_callbacks(
            checkpoint=save_path, evals=my_evals, lr=lr_base, lr_decay=lr_decay, lr_min=lr_min, lr_on_batch=lr_on_batch
        )
        self.my_evals = my_evals
        self.basic_callbacks = basic_callbacks
        self.my_hist = [ii for ii in self.basic_callbacks if isinstance(ii, myCallbacks.My_history)][0]

    def __search_embedding_layer__(self, model):
        for ii in range(1, 6):
            if model.layers[-ii].name == 'embedding':
                return -ii

    def __init_dataset__(self, type):
        if type == self.triplet:
            if self.train_ds == None or self.is_triplet_dataset == False:
                print(">>>> Init triplet dataset...")
                # batch_size = int(self.batch_size / 4 * 1.5)
                batch_size = self.batch_size // 4
                tt = data.Triplet_dataset(self.data_path, batch_size=batch_size, random_status=self.random_status, random_crop=(100, 100, 3))
                self.train_ds = tt.train_dataset
                self.classes = self.train_ds.element_spec[-1].shape[-1]
                self.is_triplet_dataset = True
        else:
            if self.train_ds == None or self.is_triplet_dataset == True:
                print(">>>> Init softmax dataset...")
                self.train_ds = data.prepare_dataset(
                    self.data_path, batch_size=self.batch_size, random_status=self.random_status, random_crop=(100, 100, 3)
                )
                self.classes = self.train_ds.element_spec[-1].shape[-1]
                self.is_triplet_dataset = False

    def __init_optimizer__(self, optimizer):
        if optimizer == None:
            if self.model != None and self.model.optimizer != None:
                # Model loaded from .h5 file already compiled
                self.optimizer = self.model.optimizer
            else:
                self.optimizer = self.default_optimizer
        else:
            self.optimizer = optimizer

    def __init_model__(self, type):
        inputs = self.basic_model.inputs[0]
        embedding = self.basic_model.outputs[0]
        while self.model is not None and isinstance(self.model.layers[-1], keras.layers.Concatenate):
            # In case of centerloss or concatenated triplet model
            self.model = keras.models.Model(inputs, self.model.layers[-2].output)

        if type == self.softmax:
            if self.model == None or self.model.output_names[-1] != self.softmax:
                print(">>>> Add softmax layer...")
                output = keras.layers.Dense(self.classes, name=self.softmax, activation="softmax")(embedding)
                self.model = keras.models.Model(inputs, output)
        elif type == self.arcface:
            if self.model == None or self.model.output_names[-1] != self.arcface:
                print(">>>> Add arcface layer...")
                output = NormDense(self.classes, name=self.arcface)(embedding)
                self.model = keras.models.Model(inputs, output)
        elif type == self.triplet:
            self.model = self.basic_model
        else:
            print("What do you want!!!")

    def __init_type_by_loss__(self, loss):
        print(">>>> Init type by loss function name...")
        if loss.__class__.__name__ == "function":
            ss = loss.__name__.lower()
        else:
            ss = loss.__class__.__name__.lower()
        if self.softmax in ss or ss == "categorical_crossentropy":
            return self.softmax
        elif self.arcface in ss:
            return self.arcface
        elif self.triplet in ss:
            return self.triplet
        else:
            return self.softmax

    def __basic_train__(self, loss, epochs, initial_epoch=0):
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=self.metrics)
        self.model.fit(
            self.train_ds,
            epochs=epochs,
            verbose=1,
            callbacks=self.callbacks,
            initial_epoch=initial_epoch,
            steps_per_epoch=self.steps_per_epoch,
            use_multiprocessing=True,
            workers=4,
        )

    def logits_accuracy(self, y_true, y_pred):
        """ Accuracy function for logits only """
        # tf.print(y_true.shape, y_pred.shape)
        # labels = tf.one_hot(tf.squeeze(y_true), depth=classes, dtype=tf.int32)
        return tf.keras.metrics.categorical_accuracy(y_true, y_pred[:, -self.classes :])

    def train(self, train_schedule, initial_epoch=0):
        for sch in train_schedule:
            if sch.get("loss", None) is None:
                continue
            cur_loss = sch["loss"]
            self.basic_model.trainable = True
            self.__init_optimizer__(sch.get("optimizer", None))

            if isinstance(cur_loss, losses.TripletLossWapper) and cur_loss.logits_loss is not None:
                type = sch.get("type", None) or self.__init_type_by_loss__(cur_loss.logits_loss)
                cur_loss.feature_dim = self.basic_model.output_shape[-1]
                print(">>>> Train Triplet + %s, feature_dim = %d ..." % (type, cur_loss.feature_dim))
                self.__init_dataset__(self.triplet)
                self.__init_model__(type)
                self.model = keras.models.Model(
                    self.model.inputs[0], keras.layers.concatenate([self.basic_model.outputs[0], self.model.outputs[-1]])
                )
                type = self.triplet + " + " + type
            else:
                type = sch.get("type", None) or self.__init_type_by_loss__(cur_loss)
                print(">>>> Train %s..." % type)
                self.__init_dataset__(type)
                self.__init_model__(type)

            if sch.get("centerloss", False):
                print(">>>> Train centerloss...")
                center_loss = cur_loss
                if not isinstance(center_loss, losses.CenterLoss):
                    feature_dim = self.basic_model.output_shape[-1]
                    # initial_file = self.basic_model.name + "_centers.npy"
                    initial_file = os.path.splitext(self.save_path)[0] + "_centers.npy"
                    logits_loss = cur_loss
                    center_loss = losses.CenterLoss(
                        self.classes, feature_dim=feature_dim, factor=1.0, initial_file=initial_file, logits_loss=logits_loss
                    )
                    cur_loss = center_loss
                self.model = keras.models.Model(
                    self.model.inputs[0], keras.layers.concatenate([self.basic_model.outputs[0], self.model.outputs[-1]])
                )
                self.callbacks = self.my_evals + [center_loss.save_centers_callback] + self.basic_callbacks
            else:
                self.callbacks = self.my_evals + self.basic_callbacks
            self.metrics = None if type == self.triplet else [self.logits_accuracy]

            if sch.get("bottleneckOnly", False):
                print(">>>> Train bottleneckOnly...")
                self.basic_model.trainable = False
                self.callbacks = self.callbacks[len(self.my_evals) :]  # Exclude evaluation callbacks
                self.__basic_train__(cur_loss, sch["epoch"], initial_epoch=0)
                self.basic_model.trainable = True
            else:
                self.__basic_train__(cur_loss, initial_epoch + sch["epoch"], initial_epoch=initial_epoch)
                initial_epoch += sch["epoch"]

            print(
                ">>>> Train %s DONE!!! epochs = %s, model.stop_training = %s"
                % (type, self.model.history.epoch, self.model.stop_training)
            )
            print(">>>> My history:")
            self.my_hist.print_hist()
            if self.model.stop_training == True:
                print(">>>> But it's an early stop, break...")
                break
            print()
