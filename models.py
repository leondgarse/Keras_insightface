import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def print_buildin_models():
    print(
        """
    >>>> buildin_models
    MXNet version resnet: mobilenet_m1, r34, r50, r100, r101,
    Keras application: mobilenet, mobilenetv2, resnet50, resnet50v2, resnet101, resnet101v2, resnet152, resnet152v2
    EfficientNet: efficientnetb[0-7], efficientnetl2,
    Custom 1: ghostnet, mobilefacenet, mobilenetv3_small, mobilenetv3_large, se_mobilefacenet
    Custom 2: botnet50, botnet101, botnet152, resnest50, resnest101, se_resnext
    Or other names from keras.applications like DenseNet121 / InceptionV3 / NASNetMobile / VGG19.
    """,
        end="",
    )


# MXNET: bn_momentum=0.9, bn_epsilon=2e-5, TF default: bn_momentum=0.99, bn_epsilon=0.001, PyTorch default: momentum=0.1, eps=1e-05
# MXNET: use_bias=True, scale=False, cavaface.pytorch: use_bias=False, scale=True
def buildin_models(
    name,
    dropout=1,
    emb_shape=512,
    input_shape=(112, 112, 3),
    output_layer="GDC",
    bn_momentum=0.99,
    bn_epsilon=0.001,
    add_pointwise_conv=False,
    use_bias=False,
    scale=True,
    weights="imagenet",
    **kwargs
):
    name_lower = name.lower()
    """ Basic model """
    if name_lower == "mobilenet":
        xx = keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights=weights, **kwargs)
    elif name_lower == "mobilenet_m1":
        from backbones import mobilenet

        xx = mobilenet.MobileNet(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name_lower == "mobilenetv2":
        xx = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights, **kwargs)
    elif name_lower == "r34" or name_lower == "r50" or name_lower == "r100" or name_lower == "r101":
        from backbones import resnet  # MXNet insightface version resnet

        model_name = "ResNet" + name_lower[1:]
        model_class = getattr(resnet, model_name)
        xx = model_class(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name_lower.startswith("resnet"):  # keras.applications.ResNetxxx
        if name_lower.endswith("v2"):
            model_name = "ResNet" + name_lower[len("resnet") : -2] + "V2"
        else:
            model_name = "ResNet" + name_lower[len("resnet") :]
        model_class = getattr(keras.applications, model_name)
        xx = model_class(weights=weights, include_top=False, input_shape=input_shape, **kwargs)
    elif name_lower.startswith("efficientnet"):
        # import tensorflow.keras.applications.efficientnet as efficientnet
        from backbones import efficientnet

        model_name = "EfficientNet" + name_lower[-2:].upper()
        model_class = getattr(efficientnet, model_name)
        xx = model_class(weights=weights, include_top=False, input_shape=input_shape, **kwargs)  # or weights='imagenet'
    elif name_lower.startswith("se_resnext"):
        from keras_squeeze_excite_network import se_resnext

        if name_lower.endswith("101"):  # se_resnext101
            depth = [3, 4, 23, 3]
        else:  # se_resnext50
            depth = [3, 4, 6, 3]
        xx = se_resnext.SEResNextImageNet(weights=weights, input_shape=input_shape, include_top=False, depth=depth)
    elif name_lower.startswith("resnest"):
        from backbones import resnest

        if name_lower == "resnest50":
            xx = resnest.ResNest50(input_shape=input_shape)
        else:
            xx = resnest.ResNest101(input_shape=input_shape)
    elif name_lower.startswith("mobilenetv3"):
        from backbones import mobilenet_v3

        # from backbones import mobilenetv3 as mobilenet_v3
        model_class = mobilenet_v3.MobileNetV3Small if "small" in name_lower else mobilenet_v3.MobileNetV3Large
        # from tensorflow.keras.layers.experimental.preprocessing import Rescaling
        # model_class = keras.applications.MobileNetV3Small if "small" in name_lower else keras.applications.MobileNetV3Large
        xx = model_class(input_shape=input_shape, include_top=False, weights=weights)
        # xx = keras.models.clone_model(xx, clone_function=lambda layer: Rescaling(1.) if isinstance(layer, Rescaling) else layer)
    elif "mobilefacenet" in name_lower or "mobile_facenet" in name_lower:
        from backbones import mobile_facenet

        use_se = True if "se" in name_lower else False
        xx = mobile_facenet.mobile_facenet(input_shape=input_shape, include_top=False, name=name, use_se=use_se)
    elif name_lower == "ghostnet":
        from backbones import ghost_model

        xx = ghost_model.GhostNet(input_shape=input_shape, include_top=False, width=1.3)
    elif name_lower.startswith("botnet"):
        from backbones import botnet

        model_name = "BotNet" + name_lower[len("botnet") :]
        model_class = getattr(botnet, model_name)
        xx = model_class(include_top=False, input_shape=input_shape, strides=1, **kwargs)
    elif hasattr(keras.applications, name):
        model_class = getattr(keras.applications, name)
        xx = model_class(weights=weights, include_top=False, input_shape=input_shape)
    else:
        return None
    xx.trainable = True

    if bn_momentum != 0.99 or bn_epsilon != 0.001:
        print(">>>> Change BatchNormalization momentum and epsilon default value.")
        for ii in xx.layers:
            if isinstance(ii, keras.layers.BatchNormalization):
                ii.momentum, ii.epsilon = bn_momentum, bn_epsilon

    inputs = xx.inputs[0]
    nn = xx.outputs[0]

    if add_pointwise_conv:  # Model using `pointwise_conv + GDC` / `pointwise_conv + E` is smaller than `E`
        nn = keras.layers.Conv2D(512, 1, use_bias=False, padding="same")(nn)
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
        nn = keras.layers.PReLU(shared_axes=[1, 2])(nn)

    if output_layer == "E":
        """ Fully Connected """
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Flatten()(nn)
        nn = keras.layers.Dense(emb_shape, activation=None, use_bias=use_bias, kernel_initializer="glorot_normal")(nn)
    else:
        """ GDC """
        nn = keras.layers.DepthwiseConv2D(int(nn.shape[1]), depth_multiplier=1, use_bias=False)(nn)
        # nn = keras.layers.Conv2D(512, int(nn.shape[1]), use_bias=False, padding="valid", groups=512)(nn)
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Conv2D(emb_shape, 1, use_bias=use_bias, activation=None, kernel_initializer="glorot_normal")(nn)
        nn = keras.layers.Flatten()(nn)
        # nn = keras.layers.Dense(emb_shape, activation=None, use_bias=True, kernel_initializer="glorot_normal")(nn)

    # `fix_gamma=True` in MXNet means `scale=False` in Keras
    embedding = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, scale=scale)(nn)
    embedding_fp32 = keras.layers.Activation("linear", dtype="float32", name="embedding")(embedding)
    basic_model = keras.models.Model(inputs, embedding_fp32, name=xx.name)
    return basic_model


def add_l2_regularizer_2_model(model, weight_decay, custom_objects={}, apply_to_batch_normal=True):
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
            if layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.DepthwiseConv2D):
            # print(">>>> DepthwiseConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["depthwise_regularizer"]
            if layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.SeparableConv2D):
            # print(">>>> SeparableConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["pointwise_regularizer", "depthwise_regularizer"]
            if layer.use_bias:
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
            print(">>>> Convert ReLU:", layer.name)
            if target_activation == "PReLU":
                layer_name = layer.name.replace("_relu", "_prelu")
                return PReLU(shared_axes=[1, 2], name=layer_name, **kwargs)
            if target_activation == "swish":
                layer_name = layer.name.replace("_relu", "_swish")
                return Activation(activation="swish", name=layer_name, **kwargs)
            else:
                return target_activation(**kwargs)
        return layer

    return keras.models.clone_model(model, clone_function=convert_ReLU)


class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, kernel_regularizer=None, loss_top_k=1, **kwargs):
        super(NormDense, self).__init__(**kwargs)
        self.init = keras.initializers.glorot_normal()
        self.units, self.loss_top_k = units, loss_top_k
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.supports_masking = True

    def build(self, input_shape):
        self.w = self.add_weight(
            name="norm_dense_w",
            shape=(input_shape[-1], self.units * self.loss_top_k),
            initializer=self.init,
            trainable=True,
            regularizer=self.kernel_regularizer,
        )
        super(NormDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        norm_w = K.l2_normalize(self.w, axis=0)
        inputs = K.l2_normalize(inputs, axis=1)
        output = K.dot(inputs, norm_w)
        if self.loss_top_k > 1:
            output = K.reshape(output, (-1, self.units, self.loss_top_k))
            output = K.max(output, axis=2)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(NormDense, self).get_config()
        config.update(
            {
                "units": self.units,
                "loss_top_k": self.loss_top_k,
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
