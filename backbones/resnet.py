import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation."""
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name + "bn",
    )(inputs)
    if activation:
        act_name = name + activation
        if activation.lower() == "prelu":
            nn = keras.layers.PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25), name=act_name)(nn)
        else:
            nn = keras.layers.Activation(activation=activation, name=act_name)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name=""):
    if padding.upper() == "SAME":
        inputs = keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(inputs)
    return keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="VALID",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "conv",
    )(inputs)


def se_module(inputs, se_ratio=4, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = filters // se_ratio
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = conv2d_no_bias(se, reduction, 1, name=name + "_1_")
    se = keras.layers.Activation("relu")(se)
    se = conv2d_no_bias(se, filters, 1, name=name + "_2_")
    se = keras.layers.Activation("sigmoid")(se)
    return keras.layers.Multiply()([inputs, se])


def block(inputs, out_channel, strides=1, activation="relu", use_se=False, conv_shortcut=False, name=""):
    if conv_shortcut:
        shortcut = conv2d_no_bias(inputs, out_channel, 1, strides=strides, name=name + "_shortcut_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, name=name + "_shortcut_")
    else:
        shortcut = inputs if strides == 1 else keras.layers.MaxPooling2D(1, strides=strides)(inputs)

    nn = batchnorm_with_activation(inputs, activation=None, name=name + "_1_")
    nn = conv2d_no_bias(nn, out_channel, 3, strides=1, padding="same", name=name + "_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "_2_")
    nn = conv2d_no_bias(nn, out_channel, 3, strides=strides, padding="same", name=name + "_2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "_3_")
    if use_se:
        nn = se_module(nn, se_ratio=16, name=name + "_se")
    return keras.layers.Add(name=name + "_add")([shortcut, nn])


def resnet_stack_fn(inputs, out_channels, depthes, use_se=False, use_max_pool=False, strides=2, activation="relu"):
    nn = inputs
    use_ses = use_se if isinstance(use_se, (list, tuple)) else [use_se] * len(out_channels)
    for id, (out_channel, depth, use_se) in enumerate(zip(out_channels, depthes, use_ses)):
        name = "stack" + str(id + 1)
        conv_shortcut = False if use_max_pool and inputs.shape[-1] == out_channel else True
        # print(f"{conv_shortcut = }, {use_max_pool = }")
        nn = block(nn, out_channel, strides, activation, use_se, conv_shortcut, name=name + "_block1")
        for ii in range(2, depth + 1):
            nn = block(nn, out_channel, 1, activation, use_se, False, name=name + "_block" + str(ii))
    return nn


def ResNet(input_shape, stack_fn, classes=1000, activation="relu", model_name="resnet", **kwargs):
    img_input = keras.layers.Input(shape=input_shape)
    nn = conv2d_no_bias(img_input, 64, 3, strides=1, padding="SAME", name="0_")
    nn = batchnorm_with_activation(nn, activation=activation, name="0_")

    nn = stack_fn(nn)

    if classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(classes, activation="softmax", name="predictions")(nn)

    model = tf.keras.models.Model(img_input, nn, name=model_name)
    return model


def ResNet18(input_shape, classes=1000, activation="relu", use_se=False, use_max_pool=False, model_name="ResNet18", **kwargs):
    out_channels = [64, 128, 256, 512]
    depthes = [2, 2, 2, 2]
    stack_fn = lambda nn: resnet_stack_fn(nn, out_channels, depthes, use_se, use_max_pool, activation=activation)
    return ResNet(input_shape, stack_fn, classes, activation, model_name=model_name, **kwargs)


def ResNet34(input_shape, classes=1000, activation="relu", use_se=False, use_max_pool=False, model_name="ResNet34", **kwargs):
    out_channels = [64, 128, 256, 512]
    depthes = [3, 4, 6, 3]
    stack_fn = lambda nn: resnet_stack_fn(nn, out_channels, depthes, use_se, use_max_pool, activation=activation)
    return ResNet(input_shape, stack_fn, classes, activation, model_name=model_name, **kwargs)


def ResNet50(input_shape, classes=1000, activation="relu", use_se=False, use_max_pool=False, model_name="ResNet50", **kwargs):
    out_channels = [64, 128, 256, 512]
    depthes = [3, 4, 14, 3]
    stack_fn = lambda nn: resnet_stack_fn(nn, out_channels, depthes, use_se, use_max_pool, activation=activation)
    return ResNet(input_shape, stack_fn, classes, activation, model_name=model_name, **kwargs)


def ResNet100(input_shape, classes=1000, activation="relu", use_se=False, use_max_pool=False, model_name="ResNet100", **kwargs):
    out_channels = [64, 128, 256, 512]
    depthes = [3, 13, 30, 3]
    stack_fn = lambda nn: resnet_stack_fn(nn, out_channels, depthes, use_se, use_max_pool, activation=activation)
    return ResNet(input_shape, stack_fn, classes, activation, model_name=model_name, **kwargs)


def ResNet101(input_shape, classes=1000, activation="relu", use_se=False, use_max_pool=False, model_name="ResNet101", **kwargs):
    out_channels = [64, 128, 256, 512]
    depthes = [3, 4, 23, 3]
    stack_fn = lambda nn: resnet_stack_fn(nn, out_channels, depthes, use_se, use_max_pool, activation=activation)
    return ResNet(input_shape, stack_fn, classes, activation, model_name=model_name, **kwargs)
