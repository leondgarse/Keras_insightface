import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def print_buildin_models():
    print(
        """
    >>>> buildin_models
    MXNet version resnet: mobilenet_m1, r18, r34, r50, r100, r101, se_r34, se_r50, se_r100
    Keras application: mobilenet, mobilenetv2, resnet50, resnet50v2, resnet101, resnet101v2, resnet152, resnet152v2
    EfficientNet: efficientnetb[0-7], efficientnetl2, efficientnetv2b[1-3], efficientnetv2[t, s, m, l, xl]
    Custom 1: ghostnet, mobilefacenet, mobilenetv3_small, mobilenetv3_large, se_mobilefacenet, se_resnext, vargface
    Or other names from keras.applications like DenseNet121 / InceptionV3 / NASNetMobile / VGG19.
    """,
        end="",
    )


def __init_model_from_name__(name, input_shape=(112, 112, 3), weights="imagenet", **kwargs):
    name_lower = name.lower()
    """ Basic model """
    if name_lower == "mobilenet":
        xx = keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights=weights, **kwargs)
    elif name_lower == "mobilenet_m1":
        from backbones import mobilenet_m1

        xx = mobilenet_m1.MobileNet(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name_lower == "mobilenetv2":
        xx = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights, **kwargs)
    elif "r18" in name_lower or "r34" in name_lower or "r50" in name_lower or "r100" in name_lower or "r101" in name_lower:
        from backbones import resnet  # MXNet insightface version resnet

        use_se = True if name_lower.startswith("se_") else False
        model_name = "ResNet" + name_lower[4:] if use_se else "ResNet" + name_lower[1:]
        use_se = kwargs.pop("use_se", use_se)
        model_class = getattr(resnet, model_name)
        xx = model_class(input_shape=input_shape, classes=0, use_se=use_se, model_name=model_name, **kwargs)
    elif name_lower.startswith("resnet"):  # keras.applications.ResNetxxx
        if name_lower.endswith("v2"):
            model_name = "ResNet" + name_lower[len("resnet") : -2] + "V2"
        else:
            model_name = "ResNet" + name_lower[len("resnet") :]
        model_class = getattr(keras.applications, model_name)
        xx = model_class(weights=weights, include_top=False, input_shape=input_shape, **kwargs)
    elif name_lower.startswith("efficientnetv2"):
        import keras_efficientnet_v2

        model_name = "EfficientNetV2" + name_lower[len("EfficientNetV2") :].upper()
        model_class = getattr(keras_efficientnet_v2, model_name)
        xx = model_class(pretrained=weights, num_classes=0, input_shape=input_shape, **kwargs)
    elif name_lower.startswith("efficientnet"):
        import keras_efficientnet_v2

        model_name = "EfficientNetV1" + name_lower[-2:].upper()
        model_class = getattr(keras_efficientnet_v2, model_name)
        xx = model_class(pretrained=weights, num_classes=0, input_shape=input_shape, **kwargs)
    elif name_lower.startswith("se_resnext"):
        from keras_squeeze_excite_network import se_resnext

        if name_lower.endswith("101"):  # se_resnext101
            depth = [3, 4, 23, 3]
        else:  # se_resnext50
            depth = [3, 4, 6, 3]
        xx = se_resnext.SEResNextImageNet(weights=weights, input_shape=input_shape, include_top=False, depth=depth)
    elif name_lower.startswith("mobilenetv3"):
        model_class = keras.applications.MobileNetV3Small if "small" in name_lower else keras.applications.MobileNetV3Large
        xx = model_class(input_shape=input_shape, include_top=False, weights=weights, include_preprocessing=False)
    elif "mobilefacenet" in name_lower or "mobile_facenet" in name_lower:
        from backbones import mobile_facenet

        use_se = True if "se" in name_lower else False
        xx = mobile_facenet.MobileFaceNet(input_shape=input_shape, include_top=False, name=name, use_se=use_se)
    elif name_lower == "ghostnet":
        from backbones import ghost_model

        xx = ghost_model.GhostNet(input_shape=input_shape, include_top=False, width=1.3, **kwargs)
    elif name_lower == "vargface":
        from backbones import vargface

        xx = vargface.VargFace(input_shape=input_shape, **kwargs)
    elif hasattr(keras.applications, name):
        model_class = getattr(keras.applications, name)
        xx = model_class(weights=weights, include_top=False, input_shape=input_shape, **kwargs)
    else:
        return None
    xx.trainable = True
    return xx


# MXNET: bn_momentum=0.9, bn_epsilon=2e-5, TF default: bn_momentum=0.99, bn_epsilon=0.001, PyTorch default: momentum=0.1, eps=1e-05
# MXNET: use_bias=True, scale=False, cavaface.pytorch: use_bias=False, scale=True
def buildin_models(
    stem_model,
    dropout=1,
    emb_shape=512,
    input_shape=(112, 112, 3),
    output_layer="GDC",
    bn_momentum=0.99,
    bn_epsilon=0.001,
    add_pointwise_conv=False,
    pointwise_conv_act="relu",
    use_bias=False,
    scale=True,
    weights="imagenet",
    **kwargs
):
    if isinstance(stem_model, str):
        xx = __init_model_from_name__(stem_model, input_shape, weights, **kwargs)
        name = stem_model
    else:
        name = stem_model.name
        xx = stem_model

    if bn_momentum != 0.99 or bn_epsilon != 0.001:
        print(">>>> Change BatchNormalization momentum and epsilon default value.")
        for ii in xx.layers:
            if isinstance(ii, keras.layers.BatchNormalization):
                ii.momentum, ii.epsilon = bn_momentum, bn_epsilon
        xx = keras.models.clone_model(xx)

    inputs = xx.inputs[0]
    nn = xx.outputs[0]

    if add_pointwise_conv:  # Model using `pointwise_conv + GDC` / `pointwise_conv + E` is smaller than `E`
        filters = nn.shape[-1] // 2 if add_pointwise_conv == -1 else 512  # Compitable with previous models...
        nn = keras.layers.Conv2D(filters, 1, use_bias=False, padding="valid", name="pw_conv")(nn)
        # nn = keras.layers.Conv2D(nn.shape[-1] // 2, 1, use_bias=False, padding="valid", name="pw_conv")(nn)
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="pw_bn")(nn)
        if pointwise_conv_act.lower() == "prelu":
            nn = keras.layers.PReLU(shared_axes=[1, 2], name="pw_" + pointwise_conv_act)(nn)
        else:
            nn = keras.layers.Activation(pointwise_conv_act, name="pw_" + pointwise_conv_act)(nn)

    if output_layer == "E":
        """Fully Connected"""
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="E_batchnorm")(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Flatten(name="E_flatten")(nn)
        nn = keras.layers.Dense(emb_shape, use_bias=use_bias, kernel_initializer="glorot_normal", name="E_dense")(nn)
    elif output_layer == "GAP":
        """GlobalAveragePooling2D"""
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="GAP_batchnorm")(nn)
        nn = keras.layers.GlobalAveragePooling2D(name="GAP_pool")(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Dense(emb_shape, use_bias=use_bias, kernel_initializer="glorot_normal", name="GAP_dense")(nn)
    elif output_layer == "GDC":
        """GDC"""
        nn = keras.layers.DepthwiseConv2D(nn.shape[1], use_bias=False, name="GDC_dw")(nn)
        # nn = keras.layers.Conv2D(nn.shape[-1], nn.shape[1], use_bias=False, padding="valid", groups=nn.shape[-1])(nn)
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="GDC_batchnorm")(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Conv2D(emb_shape, 1, use_bias=use_bias, kernel_initializer="glorot_normal", name="GDC_conv")(nn)
        nn = keras.layers.Flatten(name="GDC_flatten")(nn)
        # nn = keras.layers.Dense(emb_shape, activation=None, use_bias=use_bias, kernel_initializer="glorot_normal", name="GDC_dense")(nn)
    elif output_layer == "F":
        """F, E without first BatchNormalization"""
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Flatten(name="F_flatten")(nn)
        nn = keras.layers.Dense(emb_shape, use_bias=use_bias, kernel_initializer="glorot_normal", name="F_dense")(nn)

    # `fix_gamma=True` in MXNet means `scale=False` in Keras
    embedding = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, scale=scale, name="pre_embedding")(nn)
    embedding_fp32 = keras.layers.Activation("linear", dtype="float32", name="embedding")(embedding)

    basic_model = keras.models.Model(inputs, embedding_fp32, name=xx.name)
    return basic_model


@keras.utils.register_keras_serializable(package="keras_insightface")
class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, kernel_regularizer=None, loss_top_k=1, append_norm=False, partial_fc_split=0, **kwargs):
        super(NormDense, self).__init__(**kwargs)
        # self.init = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
        self.init = keras.initializers.glorot_normal()
        # self.init = keras.initializers.TruncatedNormal(mean=0, stddev=0.01)
        self.units, self.loss_top_k, self.append_norm, self.partial_fc_split = units, loss_top_k, append_norm, partial_fc_split
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.supports_masking = False

    def build(self, input_shape):
        if self.partial_fc_split > 1:
            self.cur_id = self.add_weight(name="cur_id", shape=(), initializer="zeros", dtype="int64", trainable=False)
            self.sub_weights = self.add_weight(
                name="norm_dense_w_subs",
                shape=(self.partial_fc_split, input_shape[-1], self.units * self.loss_top_k),
                initializer=self.init,
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
        else:
            self.w = self.add_weight(
                name="norm_dense_w",
                shape=(input_shape[-1], self.units * self.loss_top_k),
                initializer=self.init,
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
        super(NormDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # tf.print("tf.reduce_mean(self.w):", tf.reduce_mean(self.w))
        if self.partial_fc_split > 1:
            # self.sub_weights.scatter_nd_update([[(self.cur_id - 1) % self.partial_fc_split]], [self.w])
            # self.w.assign(tf.gather(self.sub_weights, self.cur_id))
            self.w = tf.gather(self.sub_weights, self.cur_id)
            self.cur_id.assign((self.cur_id + 1) % self.partial_fc_split)

        norm_w = tf.nn.l2_normalize(self.w, axis=0, epsilon=1e-5)
        norm_inputs = tf.nn.l2_normalize(inputs, axis=1, epsilon=1e-5)
        output = K.dot(norm_inputs, norm_w)
        if self.loss_top_k > 1:
            output = K.reshape(output, (-1, self.units, self.loss_top_k))
            output = K.max(output, axis=2)
        if self.append_norm:
            # Keep norm value low by * -1, so will not affect accuracy metrics.
            output = tf.concat([output, tf.norm(inputs, axis=1, keepdims=True) * -1], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(NormDense, self).get_config()
        config.update(
            {
                "units": self.units,
                "loss_top_k": self.loss_top_k,
                "append_norm": self.append_norm,
                "partial_fc_split": self.partial_fc_split,
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package="keras_insightface")
class NormDenseVPL(NormDense):
    def __init__(self, batch_size, units=1000, kernel_regularizer=None, vpl_lambda=0.15, start_iters=8000, allowed_delta=200, **kwargs):
        super().__init__(units, kernel_regularizer, **kwargs)
        self.vpl_lambda, self.batch_size = vpl_lambda, batch_size  # Need the actual batch_size here, for storing inputs
        # self.start_iters, self.allowed_delta = 8000 * 128 // batch_size, 200 * 128 // batch_size # adjust according to batch_size
        self.start_iters, self.allowed_delta = start_iters, allowed_delta
        # print(">>>> [NormDenseVPL], vpl_lambda={}, start_iters={}, allowed_delta={}".format(vpl_lambda, start_iters, allowed_delta))

    def build(self, input_shape):
        # self.queue_features in same shape format as self.norm_features, for easier calling tf.tensor_scatter_nd_update
        self.norm_features = self.add_weight(name="norm_features", shape=(self.batch_size, input_shape[-1]), dtype=self.compute_dtype, trainable=False)
        self.queue_features = self.add_weight(name="queue_features", shape=(self.units, input_shape[-1]), initializer=self.init, trainable=False)
        self.queue_iters = self.add_weight(name="queue_iters", shape=(self.units,), initializer="zeros", dtype="int64", trainable=False)
        self.zero_queue_lambda = tf.zeros((self.units,), dtype=self.compute_dtype)
        self.iters = self.add_weight(name="iters", shape=(), initializer="zeros", dtype="int64", trainable=False)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # tf.print("tf.reduce_mean(self.w):", tf.reduce_mean(self.w))
        self.iters.assign_add(1)
        queue_lambda = tf.cond(
            self.iters > self.start_iters,
            lambda: tf.where(self.iters - self.queue_iters <= self.allowed_delta, self.vpl_lambda, 0.0),  # prepare_queue_lambda
            lambda: self.zero_queue_lambda,
        )
        tf.print(" - vpl_sample_ratio:", tf.reduce_mean(tf.cast(queue_lambda > 0, "float32")), end="")
        # self.queue_lambda = queue_lambda

        if self.partial_fc_split > 1:
            self.w = tf.gather(self.sub_weights, self.cur_id)
            self.cur_id.assign((self.cur_id + 1) % self.partial_fc_split)

        norm_w = K.l2_normalize(self.w, axis=0)
        injected_weight = norm_w * (1 - queue_lambda) + tf.transpose(self.queue_features) * queue_lambda
        injected_norm_weight = K.l2_normalize(injected_weight, axis=0)

        # set_queue needs actual input labels, it's done in callback VPLUpdateQueue

        norm_inputs = K.l2_normalize(inputs, axis=1)
        self.norm_features.assign(norm_inputs)
        output = K.dot(norm_inputs, injected_norm_weight)
        if self.append_norm:
            # Keep norm value low by * -1, so will not affect accuracy metrics.
            output = tf.concat([output, tf.norm(inputs, axis=1, keepdims=True) * -1], axis=-1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"batch_size": self.batch_size, "vpl_lambda": self.vpl_lambda})
        return config


def add_l2_regularizer_2_model(model, weight_decay, custom_objects={}, apply_to_batch_normal=False, apply_to_bias=False):
    # https://github.com/keras-team/keras/issues/2717#issuecomment-456254176
    if 0:
        regularizers_type = {}
        for layer in model.layers:
            rrs = [kk for kk in layer.__dict__.keys() if "regularizer" in kk and not kk.startswith("_")]
            if len(rrs) != 0:
                # print(layer.name, layer.__class__.__name__, rrs)
                if layer.__class__.__name__ not in regularizers_type:
                    regularizers_type[layer.__class__.__name__] = rrs
        print(regularizers_type)

    for layer in model.layers:
        attrs = []
        if isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Conv2D):
            # print(">>>> Dense or Conv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["kernel_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.DepthwiseConv2D):
            # print(">>>> DepthwiseConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["depthwise_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.SeparableConv2D):
            # print(">>>> SeparableConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["pointwise_regularizer", "depthwise_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif apply_to_batch_normal and isinstance(layer, keras.layers.BatchNormalization):
            # print(">>>> BatchNormalization", layer.name, "scale:", layer.scale, ", center:", layer.center)
            if layer.center:
                attrs.append("beta_regularizer")
            if layer.scale:
                attrs.append("gamma_regularizer")
        elif apply_to_batch_normal and isinstance(layer, keras.layers.PReLU):
            # print(">>>> PReLU", layer.name)
            attrs = ["alpha_regularizer"]

        for attr in attrs:
            if hasattr(layer, attr) and layer.trainable:
                setattr(layer, attr, keras.regularizers.L2(weight_decay / 2))

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    # temp_weight_file = "tmp_weights.h5"
    # model.save_weights(temp_weight_file)
    # out_model = keras.models.model_from_json(model.to_json(), custom_objects=custom_objects)
    # out_model.load_weights(temp_weight_file, by_name=True)
    # os.remove(temp_weight_file)
    # return out_model
    return keras.models.clone_model(model)


def replace_ReLU_with_PReLU(model, target_activation="PReLU", **kwargs):
    from tensorflow.keras.layers import ReLU, PReLU, Activation

    def convert_ReLU(layer):
        # print(layer.name)
        if isinstance(layer, ReLU) or (isinstance(layer, Activation) and layer.activation == keras.activations.relu):
            if target_activation == "PReLU":
                layer_name = layer.name.replace("_relu", "_prelu")
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                # Default initial value in mxnet and pytorch is 0.25
                return PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25), name=layer_name, **kwargs)
            elif isinstance(target_activation, str):
                layer_name = layer.name.replace("_relu", "_" + target_activation)
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                return Activation(activation=target_activation, name=layer_name, **kwargs)
            else:
                act_class_name = target_activation.__name__
                layer_name = layer.name.replace("_relu", "_" + act_class_name)
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                return target_activation(**kwargs)
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=convert_ReLU)


@keras.utils.register_keras_serializable(package="keras_insightface")
class AconC(keras.layers.Layer):
    """
    - [Github nmaac/acon](https://github.com/nmaac/acon/blob/main/acon.py)
    - [Activate or Not: Learning Customized Activation, CVPR 2021](https://arxiv.org/pdf/2009.04759.pdf)
    """

    def __init__(self, p1=1, p2=0, beta=1, **kwargs):
        super(AconC, self).__init__(**kwargs)
        self.p1_init = tf.initializers.Constant(p1)
        self.p2_init = tf.initializers.Constant(p2)
        self.beta_init = tf.initializers.Constant(beta)
        self.supports_masking = False

    def build(self, input_shape):
        self.p1 = self.add_weight(name="p1", shape=(1, 1, 1, input_shape[-1]), initializer=self.p1_init, trainable=True)
        self.p2 = self.add_weight(name="p2", shape=(1, 1, 1, input_shape[-1]), initializer=self.p2_init, trainable=True)
        self.beta = self.add_weight(name="beta", shape=(1, 1, 1, input_shape[-1]), initializer=self.beta_init, trainable=True)
        super(AconC, self).build(input_shape)

    def call(self, inputs, **kwargs):
        p1 = inputs * self.p1
        p2 = inputs * self.p2
        beta = inputs * self.beta
        return p1 * tf.nn.sigmoid(beta) + p2

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(AconC, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SAMModel(tf.keras.models.Model):
    """
    Arxiv article: [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412.pdf)
    Implementation by: [Keras SAM (Sharpness-Aware Minimization)](https://qiita.com/T-STAR/items/8c3afe3a116a8fc08429)

    Usage is same with `keras.modeols.Model`: `model = SAMModel(inputs, outputs, rho=sam_rho, name=name)`
    """

    def __init__(self, *args, rho=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = tf.constant(rho, dtype=tf.float32)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # 1st step
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        norm = tf.linalg.global_norm(gradients)
        scale = self.rho / (norm + 1e-12)
        e_w_list = []
        for v, grad in zip(trainable_vars, gradients):
            e_w = grad * scale
            v.assign_add(e_w)
            e_w_list.append(e_w)

        # 2nd step
        with tf.GradientTape() as tape:
            y_pred_adv = self(x, training=True)
            loss_adv = self.compiled_loss(y, y_pred_adv, sample_weight=sample_weight, regularization_losses=self.losses)
        gradients_adv = tape.gradient(loss_adv, trainable_vars)
        for v, e_w in zip(trainable_vars, e_w_list):
            v.assign_sub(e_w)

        # optimize
        self.optimizer.apply_gradients(zip(gradients_adv, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


def replace_add_with_stochastic_depth(model, survivals=(1, 0.8)):
    """
    - [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
    - [tfa.layers.StochasticDepth](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth)
    """
    from tensorflow_addons.layers import StochasticDepth

    add_layers = [ii.name for ii in model.layers if isinstance(ii, keras.layers.Add)]
    total_adds = len(add_layers)
    if isinstance(survivals, float):
        survivals = [survivals] * total_adds
    elif isinstance(survivals, (list, tuple)) and len(survivals) == 2:
        start, end = survivals
        survivals = [start - (1 - end) * float(ii) / total_adds for ii in range(total_adds)]
    survivals_dict = dict(zip(add_layers, survivals))

    def __replace_add_with_stochastic_depth__(layer):
        if isinstance(layer, keras.layers.Add):
            layer_name = layer.name
            new_layer_name = layer_name.replace("_add", "_stochastic_depth")
            new_layer_name = layer_name.replace("add_", "stochastic_depth_")
            survival_probability = survivals_dict[layer_name]
            if survival_probability < 1:
                print("Converting:", layer_name, "-->", new_layer_name, ", survival_probability:", survival_probability)
                return StochasticDepth(survival_probability, name=new_layer_name)
            else:
                return layer
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=__replace_add_with_stochastic_depth__)


def replace_stochastic_depth_with_add(model, drop_survival=False):
    from tensorflow_addons.layers import StochasticDepth

    def __replace_stochastic_depth_with_add__(layer):
        if isinstance(layer, StochasticDepth):
            layer_name = layer.name
            new_layer_name = layer_name.replace("_stochastic_depth", "_lambda")
            survival = layer.survival_probability
            print("Converting:", layer_name, "-->", new_layer_name, ", survival_probability:", survival)
            if drop_survival or not survival < 1:
                return keras.layers.Add(name=new_layer_name)
            else:
                return keras.layers.Lambda(lambda xx: xx[0] + xx[1] * survival, name=new_layer_name)
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=__replace_stochastic_depth_with_add__)


def convert_to_mixed_float16(model, convert_batch_norm=False):
    policy = keras.mixed_precision.Policy("mixed_float16")
    policy_config = keras.utils.serialize_keras_object(policy)
    from tensorflow.keras.layers import InputLayer, Activation
    from tensorflow.keras.activations import linear, softmax

    def do_convert_to_mixed_float16(layer):
        if not convert_batch_norm and isinstance(layer, keras.layers.BatchNormalization):
            return layer
        if isinstance(layer, InputLayer):
            return layer
        if isinstance(layer, NormDense):
            return layer
        if isinstance(layer, Activation) and layer.activation == softmax:
            return layer
        if isinstance(layer, Activation) and layer.activation == linear:
            return layer

        aa = layer.get_config()
        aa.update({"dtype": policy_config})
        bb = layer.__class__.from_config(aa)
        bb.build(layer.input_shape)
        bb.set_weights(layer.get_weights())
        return bb

    input_tensors = keras.layers.Input(model.input_shape[1:])
    mm = keras.models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_mixed_float16)
    if model.built:
        mm.compile(optimizer=model.optimizer, loss=model.compiled_loss, metrics=model.compiled_metrics)
        # mm.optimizer, mm.compiled_loss, mm.compiled_metrics = model.optimizer, model.compiled_loss, model.compiled_metrics
        # mm.built = True
    return mm


def convert_mixed_float16_to_float32(model):
    from tensorflow.keras.layers import InputLayer, Activation
    from tensorflow.keras.activations import linear

    def do_convert_to_mixed_float16(layer):
        if not isinstance(layer, InputLayer) and not (isinstance(layer, Activation) and layer.activation == linear):
            aa = layer.get_config()
            aa.update({"dtype": "float32"})
            bb = layer.__class__.from_config(aa)
            bb.build(layer.input_shape)
            bb.set_weights(layer.get_weights())
            return bb
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_mixed_float16)


def convert_to_batch_renorm(model):
    def do_convert_to_batch_renorm(layer):
        if isinstance(layer, keras.layers.BatchNormalization):
            aa = layer.get_config()
            aa.update({"renorm": True, "renorm_clipping": {}, "renorm_momentum": aa["momentum"]})
            bb = layer.__class__.from_config(aa)
            bb.build(layer.input_shape)
            bb.set_weights(layer.get_weights() + bb.get_weights()[-3:])
            return bb
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_batch_renorm)
