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
    print(
        """
    >>>> buildin_models
    mobilenet, mobilenetv2, mobilenetv3_small, mobilenetv3_large, mobilefacenet, se_mobilefacenet, nasnetmobile
    resnet50v2, resnet101v2, se_resnext, resnest50, resnest101,
    efficientnetb0, efficientnetb1, efficientnetb2, efficientnetb3, efficientnetb4, efficientnetb5, efficientnetb6, efficientnetb7,
    """,
        end="",
    )


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
            # import tensorflow.keras.applications.efficientnet as efntf
            import efficientnet.tfkeras as efntf
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
        xx = model(weights="noisy-student", include_top=False, input_shape=input_shape)  # or weights='imagenet'
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

        size = "small" if "small" in name else "large"
        xx = mobilenetv3.MobilenetV3(input_shape=input_shape, include_top=False, size=size)
    elif "mobilefacenet" in name or "mobile_facenet" in name:
        from backbones import mobile_facenet

        use_se = True if "se" in name else False
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
        lr_decay_steps=0,  # lr_decay_steps < 1 for update lr on epoch, or update on every [NUM] batches, or list for ConstantDecayScheduler
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
            }
        )
        self.model, self.basic_model, self.save_path = None, None, save_path
        if isinstance(model, str):
            if model.endswith(".h5") and os.path.exists(model):
                print(">>>> Load model from h5 file: %s..." % model)
                with keras.utils.custom_object_scope(custom_objects):
                    self.model = keras.models.load_model(model, compile=compile, custom_objects=custom_objects)
                embedding_layer = basic_model if basic_model is not None else self.__search_embedding_layer__(self.model)
                self.basic_model = keras.models.Model(self.model.inputs[0], self.model.layers[embedding_layer].output)
                # self.model.summary()
        elif isinstance(model, keras.models.Model):
            self.model = model
            embedding_layer = basic_model if basic_model is not None else self.__search_embedding_layer__(self.model)
            self.basic_model = keras.models.Model(self.model.inputs[0], self.model.layers[embedding_layer].output)
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

        self.softmax, self.arcface, self.triplet, self.center = "softmax", "arcface", "triplet", "center"

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
            checkpoint=save_path, evals=my_evals, lr=lr_base, lr_decay=lr_decay, lr_min=lr_min, lr_decay_steps=lr_decay_steps
        )
        self.my_evals = my_evals
        self.basic_callbacks = basic_callbacks
        self.my_hist = [ii for ii in self.basic_callbacks if isinstance(ii, myCallbacks.My_history)][0]
        self.custom_callbacks = []

    def __search_embedding_layer__(self, model):
        for ii in range(1, 6):
            if model.layers[-ii].name == "embedding":
                return -ii

    def __init_dataset_triplet__(self):
        if self.train_ds == None or self.is_triplet_dataset == False:
            print(">>>> Init triplet dataset...")
            # batch_size = int(self.batch_size / 4 * 1.5)
            batch_size = self.batch_size // 4
            tt = data.Triplet_dataset(
                self.data_path, batch_size=batch_size, random_status=self.random_status, random_crop=(100, 100, 3)
            )
            self.train_ds = tt.train_dataset
            self.classes = self.train_ds.element_spec[-1].shape[-1]
            self.is_triplet_dataset = True

    def __init_dataset_softmax__(self):
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
        is_multi_output = lambda mm: len(mm.outputs) != 1 or isinstance(mm.layers[-1], keras.layers.Concatenate)
        if self.model != None and is_multi_output(self.model):
            output_layer = min(len(self.basic_model.layers), len(self.model.layers) - 1)
            self.model = keras.models.Model(inputs, self.model.layers[output_layer].output)

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
        elif type == self.triplet or type == self.center:
            self.model = self.basic_model
            self.model.output_names[0] = type + "_embedding"
        else:
            print("What do you want!!!")

    def __init_type_by_loss__(self, loss):
        print(">>>> Init type by loss function name...")
        if isinstance(loss, str):
            return self.softmax

        if loss.__class__.__name__ == "function":
            ss = loss.__name__.lower()
            if self.softmax in ss:
                return self.softmax
            if self.arcface in ss:
                return self.arcface
            if self.triplet in ss:
                return self.triplet
        else:
            if isinstance(loss, losses.TripletLossWapper):
                return self.triplet
            if isinstance(loss, losses.CenterLoss):
                return self.center
            if isinstance(loss, losses.ArcfaceLoss):
                return self.arcface
        return self.softmax

    def __basic_train__(self, loss, epochs, initial_epoch=0, loss_weights=None):
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=self.metrics, loss_weights=loss_weights)
        # self.model.compile(optimizer=self.optimizer, loss=loss, metrics=self.metrics)
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

    def train(self, train_schedule, initial_epoch=0):
        for sch in train_schedule:
            if sch.get("loss", None) is None:
                continue
            cur_loss = sch["loss"]
            type = sch.get("type", None) or self.__init_type_by_loss__(cur_loss)
            print(">>>> Train %s..." % type)

            if sch.get("triplet", False) or type == self.triplet:
                self.__init_dataset_triplet__()
            else:
                self.__init_dataset_softmax__()

            self.basic_model.trainable = True
            self.__init_optimizer__(sch.get("optimizer", None))
            self.__init_model__(type)

            # loss_weights
            cur_loss = [cur_loss]
            self.callbacks = self.my_evals + self.custom_callbacks + self.basic_callbacks
            loss_weights = None
            if sch.get("centerloss", False) and type != self.center:
                print(">>>> Attach centerloss...")
                emb_shape = self.basic_model.output_shape[-1]
                initial_file = os.path.splitext(self.save_path)[0] + "_centers.npy"
                center_loss = losses.CenterLoss(self.classes, emb_shape=emb_shape, initial_file=initial_file)
                cur_loss = [center_loss, *cur_loss]
                loss_weights = {ii: 1.0 for ii in self.model.output_names}
                nns = self.model.output_names
                self.model = keras.models.Model(self.model.inputs[0], self.basic_model.outputs + self.model.outputs)
                self.model.output_names[0] = self.center + "_embedding"
                for id, nn in enumerate(nns):
                    self.model.output_names[id + 1] = nn
                self.callbacks = (
                    self.my_evals + self.custom_callbacks + [center_loss.save_centers_callback] + self.basic_callbacks
                )
                loss_weights.update({self.model.output_names[0]: float(sch["centerloss"])})

            if sch.get("triplet", False) and type != self.triplet:
                print(">>>> Attach tripletloss...")
                alpha = sch.get("alpha", 0.35)
                triplet_loss = losses.BatchHardTripletLoss(alpha=alpha)
                cur_loss = [triplet_loss, *cur_loss]
                loss_weights = loss_weights if loss_weights is not None else {ii: 1.0 for ii in self.model.output_names}
                nns = self.model.output_names
                self.model = keras.models.Model(self.model.inputs[0], self.basic_model.outputs + self.model.outputs)
                self.model.output_names[0] = self.triplet + "_embedding"
                for id, nn in enumerate(nns):
                    self.model.output_names[id + 1] = nn
                loss_weights.update({self.model.output_names[0]: float(sch["triplet"])})

            print(">>>> loss_weights:", loss_weights)
            self.metrics = {ii: None if "embedding" in ii else "accuracy" for ii in self.model.output_names}

            try:
                import tensorflow_addons as tfa
            except:
                pass
            else:
                if isinstance(self.optimizer, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension):
                    print(">>>> Insert weight decay callback...")
                    lr_base, wd_base = self.optimizer.lr.numpy(), self.optimizer.weight_decay.numpy()
                    wd_callback = myCallbacks.OptimizerWeightDecay(lr_base, wd_base)
                    self.callbacks.insert(-2, wd_callback) # should be after lr_scheduler

            if sch.get("bottleneckOnly", False):
                print(">>>> Train bottleneckOnly...")
                self.basic_model.trainable = False
                self.callbacks = self.callbacks[len(self.my_evals) :]  # Exclude evaluation callbacks
                self.__basic_train__(cur_loss, sch["epoch"], initial_epoch=0, loss_weights=loss_weights)
                self.basic_model.trainable = True
            else:
                self.__basic_train__(
                    cur_loss, initial_epoch + sch["epoch"], initial_epoch=initial_epoch, loss_weights=loss_weights
                )
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
