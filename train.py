import data
import evals
import losses
import myCallbacks
# import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import os


def buildin_models(name, dropout=1, emb_shape=512, **kwargs):
    """ Basic model """
    if name == "MobileNet":
        xx = keras.applications.MobileNet(input_shape=(112, 112, 3), include_top=False, weights=None, **kwargs)
    elif name == "MobileNetV2":
        xx = keras.applications.MobileNetV2(input_shape=(112, 112, 3), include_top=False, weights=None, **kwargs)
    elif name == "ResNet50V2":
        xx = keras.applications.ResNet50V2(input_shape=(112, 112, 3), include_top=False, weights="imagenet", **kwargs)
    elif name == "ResNet101V2":
        xx = keras.applications.ResNet101V2(input_shape=(112, 112, 3), include_top=False, weights="imagenet", **kwargs)
    elif name == "NASNetMobile":
        xx = keras.applications.NASNetMobile(input_shape=(112, 112, 3), include_top=False, weights=None, **kwargs)
    else:
        return None
    # xx = keras.models.load_model('checkpoints/mobilnet_v1_basic_922667.h5', compile=False)
    xx.trainable = True

    inputs = xx.inputs[0]
    nn = xx.outputs[0]
    nn = keras.layers.Conv2D(emb_shape, xx.output_shape[1], use_bias=False)(nn)
    # BatchNormalization(momentum=0.99, epsilon=0.001)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.PReLU(shared_axes=[1, 2])(nn)
    if dropout > 0 and dropout < 1:
        nn = keras.layers.Dropout(dropout)(nn)
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(emb_shape, activation=None, use_bias=False, kernel_initializer="glorot_normal")(nn)
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

    # def compute_output_shape(self, input_shape):
    #     shape = tf.TensorShape(input_shape).as_list()
    #     shape[-1] = self.units
    #     return tf.TensorShape(shape)

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
        eval_paths,
        save_path,
        basic_model=-2,
        model=None,
        compile=True,
        batch_size=128,
        lr_base=0.001,
        lr_decay=0.05,
        lr_min=5e-5,
        random_status=3,
        custom_objects={},
    ):
        self.model, self.basic_model = None, None
        if isinstance(model, str):
            if model.endswith(".h5") and os.path.exists(model) and isinstance(basic_model, int):
                print(">>>> Load model from h5 file: %s..." % model)
                custom_objects.update({
                    "NormDense": NormDense,
                    "ArcfaceLoss": losses.ArcfaceLoss,
                    "margin_softmax": losses.margin_softmax,
                    "arcface_loss": losses.arcface_loss,
                    "Center_loss": losses.Center_loss,
                    "batch_hard_triplet_loss": losses.batch_hard_triplet_loss,
                    "batch_all_triplet_loss": losses.batch_all_triplet_loss,
                })
                self.model = keras.models.load_model(model, compile=compile, custom_objects=custom_objects)
                self.basic_model = keras.models.Model(self.model.inputs[0], self.model.layers[basic_model].output)
                self.model.summary()
        elif isinstance(basic_model, str):
            if basic_model.endswith(".h5") and os.path.exists(basic_model):
                print(">>>> Load basic_model from h5 file: %s..." % basic_model)
                self.basic_model = keras.models.load_model(basic_model, compile=compile)
        elif isinstance(basic_model, keras.models.Model):
            self.basic_model = basic_model

        if self.basic_model == None:
            print(
                "Initialize model by:\n"
                "| basicmodel                               | model          |\n"
                "| ---------------------------------------- | -------------- |\n"
                "| model structure                          | None           |\n"
                "| basic model .h5 file                     | None           |\n"
                "| model layer index for basic model output | model .h5 file |\n"
            )
            return

        self.softmax, self.arcface, self.triplet = "softmax", "arcface", "triplet"
        self.data_path, self.batch_size, self.random_status = data_path, batch_size, random_status
        self.train_ds, self.steps_per_epoch, self.classes = None, 0, 0
        self.is_triplet_dataset = False
        self.default_optimizer = "adam"
        self.metrics = ["accuracy"]
        my_evals = [
            evals.epoch_eval_callback(self.basic_model, ii, save_model=None, eval_freq=1, flip=True) for ii in eval_paths
        ]
        my_evals[-1].save_model = os.path.splitext(save_path)[0]
        basic_callbacks = myCallbacks.basic_callbacks(checkpoint=save_path, evals=my_evals, lr=lr_base, lr_decay=lr_decay, lr_min=lr_min)
        self.my_evals = my_evals
        self.basic_callbacks = basic_callbacks
        self.my_hist = self.basic_callbacks[-2]

    def __init_dataset__(self, type):
        if type == self.triplet:
            if self.train_ds == None or self.is_triplet_dataset == False:
                print(">>>> Init triplet dataset...")
                batch_size = int(self.batch_size / 4 * 1.5)
                tt = data.Triplet_dataset(self.data_path, batch_size=batch_size, random_status=self.random_status)
                self.train_ds, self.steps_per_epoch = tt.train_dataset, tt.steps_per_epoch
                self.is_triplet_dataset = True
        else:
            if self.train_ds == None or self.is_triplet_dataset == True:
                print(">>>> Init softmax dataset...")
                self.train_ds, self.steps_per_epoch, self.classes = data.prepare_for_training(
                    self.data_path, batch_size=self.batch_size, random_status=self.random_status
                )
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

        # In case of centerloss model
        if len(self.model.outputs) != 1:
            self.model = keras.models.Model(inputs, self.model.outputs[-1])

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

    def __init_metrics_callbacks__(self, type, center_loss=None, bottleneckOnly=False):
        if center_loss:
            self.callbacks = self.my_evals + [center_loss.save_centers_callback] + self.basic_callbacks
            self.metrics = [center_loss.accuracy]
        elif type == self.triplet:
            self.callbacks = self.my_evals + self.basic_callbacks
            self.metrics = None
        else:
            self.callbacks = self.my_evals + self.basic_callbacks
            self.metrics = ["accuracy"]

        if bottleneckOnly:
            self.callbacks = self.callbacks[len(self.my_evals) :]  # Exclude evaluation callbacks

    def __basic_train__(self, loss, epochs, initial_epoch=0):
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=self.metrics)
        self.model.fit(
            self.train_ds,
            epochs=epochs,
            verbose=1,
            callbacks=self.callbacks,
            steps_per_epoch=self.steps_per_epoch,
            initial_epoch=initial_epoch,
            use_multiprocessing=True,
            workers=4,
        )

    def train(self, train_schedule, initial_epoch=0):
        for sch in train_schedule:
            type = sch.get("type", None) or self.__init_type_by_loss__(sch["loss"])
            print(">>>> Train %s..." % type)

            self.basic_model.trainable = True
            self.__init_optimizer__(sch.get("optimizer", None))
            self.__init_dataset__(type)
            self.__init_model__(type)
            if sch.get("centerloss", False):
                print(">>>> Train centerloss...")
                if type == self.triplet:
                    print(">>>> Center loss combined with triplet, skip")
                    continue
                center_loss = sch["loss"]
                if center_loss.__class__.__name__ != losses.Center_loss.__name__:
                    feature_dim = self.basic_model.output_shape[-1]
                    initial_file = self.basic_model.name + "_centers.npy"
                    logits_loss = sch["loss"]
                    center_loss = losses.Center_loss(
                        self.classes, feature_dim=feature_dim, factor=1.0, initial_file=initial_file, logits_loss=logits_loss
                    )
                    sch["loss"] = center_loss
                self.model = keras.models.Model(
                    self.model.inputs[0], keras.layers.concatenate([self.basic_model.outputs[0], self.model.outputs[-1]])
                )
            else:
                center_loss = None
            self.__init_metrics_callbacks__(type, center_loss, sch.get("bottleneckOnly", False))

            if sch.get("bottleneckOnly", False):
                print(">>>> Train bottleneckOnly...")
                self.basic_model.trainable = False
                self.__basic_train__(sch["loss"], sch["epoch"], initial_epoch=0)
                self.basic_model.trainable = True
            else:
                self.__basic_train__(sch["loss"], initial_epoch + sch["epoch"], initial_epoch=initial_epoch)
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
