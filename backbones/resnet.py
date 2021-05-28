import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.keras import backend as K

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name=""):
    return layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
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
    se = layers.Activation("relu")(se)
    se = conv2d_no_bias(se, filters, 1, name=name + "_2_")
    se = layers.Activation("sigmoid")(se)
    return layers.Multiply()([inputs, se])


def block(inputs, out_channel, strides=1, activation="relu", use_se=False, conv_shortcut=False, name=""):
    if conv_shortcut:
        shortcut = conv2d_no_bias(inputs, out_channel, 1, strides=strides, name=name + "_0_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, name=name + "_0_")
    else:
        shortcut = inputs

    nn = batchnorm_with_activation(inputs, activation=None, name=name + "_1_")
    nn = conv2d_no_bias(nn, out_channel, 3, strides=1, padding="same", name=name + "_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "_2_")
    nn = conv2d_no_bias(nn, out_channel, 3, strides=strides, padding="same", name=name + "_2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "_3_")
    if use_se:
        nn = se_module(nn, se_ratio=16, name=name + "_se")
    return layers.Add(name=name + "_add")([shortcut, nn])


def stack(inputs, out_channel, num_blocks, strides=2, activation="relu", use_se=False, name=""):
    nn = block(inputs, out_channel, strides, activation, use_se, True, name=name + "_block1")
    for ii in range(2, num_blocks + 1):
        nn = block(nn, out_channel, 1, activation, use_se, False, name=name + "_block" + str(ii))
    return nn


def ResNet(input_shape, stack_fn, classes=1000, classifier_activation="softmax", model_name="ir_resnet", **kwargs):
    img_input = layers.Input(shape=input_shape)
    nn = conv2d_no_bias(img_input, 64, 3, strides=1, padding="SAME", name="conv1_conv")
    nn = batchnorm_with_activation(nn, name="conv1_bn")

    nn = stack_fn(nn)

    if classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(classes, activation=classifier_activation, name="predictions")(nn)

    model = tf.keras.models.Model(img_input, nn, name=model_name)
    return model


def ResNet34(input_shape, classes=1000, activation="relu", use_se=False, model_name="resnet34", **kwargs):
    def stack_fn(nn):
        nn = stack(nn, 64, 3, activation=activation, use_se=use_se, name="stack2")
        nn = stack(nn, 128, 4, activation=activation, use_se=use_se, name="stack3")
        nn = stack(nn, 256, 6, activation=activation, use_se=use_se, name="stack4")
        return stack(nn, 512, 3, activation=activation, use_se=use_se, name="stack5")

    return ResNet(input_shape, stack_fn, classes, model_name=model_name, **kwargs)


def ResNet50(input_shape, classes=1000, activation="relu", use_se=False, model_name="resnet50", **kwargs):
    def stack_fn(nn):
        nn = stack(nn, 64, 3, activation=activation, use_se=use_se, name="stack2")
        nn = stack(nn, 128, 4, activation=activation, use_se=use_se, name="stack3")
        nn = stack(nn, 256, 14, activation=activation, use_se=use_se, name="stack4")
        return stack(nn, 512, 3, activation=activation, use_se=use_se, name="stack5")

    return ResNet(input_shape, stack_fn, classes, model_name=model_name, **kwargs)


def ResNet100(input_shape, classes=1000, activation="relu", use_se=False, model_name="resnet100", **kwargs):
    def stack_fn(nn):
        nn = stack(nn, 64, 3, activation=activation, use_se=use_se, name="stack2")
        nn = stack(nn, 128, 13, activation=activation, use_se=use_se, name="stack3")
        nn = stack(nn, 256, 30, activation=activation, use_se=use_se, name="stack4")
        return stack(nn, 512, 3, activation=activation, use_se=use_se, name="stack5")

    return ResNet(input_shape, stack_fn, classes, model_name=model_name, **kwargs)


def ResNet101(input_shape, classes=1000, activation="relu", use_se=False, model_name="resnet101", **kwargs):
    def stack_fn(nn):
        nn = stack(nn, 64, 3, activation=activation, use_se=use_se, name="stack2")
        nn = stack(nn, 128, 4, activation=activation, use_se=use_se, name="stack3")
        nn = stack(nn, 256, 23, activation=activation, use_se=use_se, name="stack4")
        return stack(nn, 512, 3, activation=activation, use_se=use_se, name="stack5")

    return ResNet(input_shape, stack_fn, classes, model_name=model_name, **kwargs)
