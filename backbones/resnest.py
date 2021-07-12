import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.keras import backend as K

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


def batchnorm_with_activation(inputs, activation="relu", name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name="", **kwargs):
    return layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "conv",
        **kwargs,
    )(inputs)


def rsoftmax(inputs, filters, radix, groups):
    if radix > 1:
        nn = tf.reshape(inputs, [-1, groups, radix, filters // groups])
        nn = K.softmax(nn, axis=2)
        nn = tf.reshape(nn, [-1, 1, 1, radix * filters])
    else:
        nn = layers.Activation("sigmoid")(inputs)
    return nn


def split_attention_conv2d(inputs, filters, kernel_size=3, groups=1, radix=2, activation="relu", name=""):
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]
    in_channels = inputs.shape[-1]
    logits = conv2d_no_bias(inputs, filters * radix, kernel_size, padding="same", groups=groups * radix, name=name + "1_")
    logits = batchnorm_with_activation(logits, activation=activation, name=name + "1_")

    if radix > 1:
        splited = tf.split(logits, radix, axis=-1)
        gap = tf.reduce_sum(splited, axis=0)
    else:
        gap = logits
    gap = tf.reduce_mean(gap, [h_axis, w_axis], keepdims=True)

    reduction_factor = 4
    inter_channels = max(in_channels * radix // reduction_factor, 32)
    atten = layers.Conv2D(inter_channels, kernel_size=1, name=name + "2_")(gap)
    atten = batchnorm_with_activation(atten, activation=activation, name=name + "2_")
    atten = layers.Conv2D(filters * radix, kernel_size=1, name=name + "3_")(atten)
    atten = rsoftmax(atten, filters, radix, groups)
    out = layers.Multiply()([atten, logits])

    if radix > 1:
        out = tf.split(out, radix, axis=-1)
        out = tf.reduce_sum(out, axis=0)
    return out


def block(inputs, filters, strides=1, activation="relu", use_se=False, radix=2, name=""):
    if strides != 1 or inputs.shape[-1] != filters * 4:
        short_cut = layers.AveragePooling2D(pool_size=strides, strides=strides, padding="same", name=name + "st_pool")(inputs)
        short_cut = conv2d_no_bias(short_cut, filters * 4, kernel_size=1, padding="same", name=name + "shortcut_")
        short_cut = batchnorm_with_activation(short_cut, activation=None, name=name + "_shortcut_")
    else:
        short_cut = inputs

    nn = conv2d_no_bias(inputs, filters, 1, padding="same", name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")

    nn = split_attention_conv2d(nn, filters=filters, kernel_size=3, groups=1, radix=radix, name=name + "sa_")
    if strides > 1:
        nn = layers.AveragePooling2D(pool_size=3, strides=strides, padding="same", name=name + "pool_")(nn)
    nn = conv2d_no_bias(nn, filters * 4, 1, padding="same", name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")

    nn = layers.Add()([short_cut, nn])
    nn = layers.Activation(activation)(nn)
    return nn


def stack(inputs, blocks, filters, strides, radix=2, name=""):
    nn = block(inputs, filters=filters, strides=strides, radix=radix, name=name+"block1_")
    for ii in range(2, blocks + 1):
        nn = block(nn, filters=filters, strides=1, radix=radix, name=name+"block{}_".format(ii))
    return nn


def stem(inputs, stem_width, activation="relu", deep_stem=False, name=""):
    if deep_stem:
        nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, padding="same", name=name+"1_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name+"1_")
        nn = conv2d_no_bias(nn, stem_width, 3, strides=1, padding="same", name=name+"2_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name+"2_")
        nn = conv2d_no_bias(nn, stem_width * 2, 3, strides=1, padding="same", name=name+"3_")
    else:
        nn = conv2d_no_bias(inputs, stem_width, 7, strides=2, padding="same", name=name+"3_")

    nn = batchnorm_with_activation(nn, activation=activation, name=name+"3_")
    nn = layers.MaxPool2D(pool_size=3, strides=2, padding="same", name=name+"pool")(nn)
    return nn


def ResNest(input_shape, blocks_set=[3, 4, 6, 3], stem_width=32, classes=1000, activation="relu", radix=2, model_name="resnest", **kwargs):
    img_input = layers.Input(shape=input_shape)
    nn = stem(img_input, stem_width, deep_stem=True, name="stem_")

    nn = stack(nn, blocks=blocks_set[0], filters=64, strides=1, radix=radix, name="stack1_")
    nn = stack(nn, blocks=blocks_set[1], filters=128, strides=2, radix=radix, name="stack2_")
    nn = stack(nn, blocks=blocks_set[2], filters=256, strides=2, radix=radix, name="stack3_")
    nn = stack(nn, blocks=blocks_set[3], filters=512, strides=2, radix=radix, name="stack4_")

    if classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(classes, activation="softmax", name="predictions")(nn)

    model = tf.keras.models.Model(img_input, nn, name=model_name)
    return model


def ResNest50(input_shape=(112, 112, 3), stem_width=32, classes=1000, activation="relu", radix=2, model_name="ResNest50", **kwargs):
    return ResNest(blocks_set=[3, 4, 6, 3], **locals())


def ResNest101(input_shape=(112, 112, 3), stem_width=64, classes=1000, activation="relu", radix=2, model_name="ResNest101", **kwargs):
    return ResNest(blocks_set=[3, 4, 23, 3], **locals())
