import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

"""Building Block Functions"""

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


def hard_swish(inputs, name=None):
    """`out = xx * relu6(xx + 3) / 6`, arxiv: https://arxiv.org/abs/1905.02244"""
    return keras.layers.Multiply(name=name)([inputs, tf.nn.relu6(inputs + 3) / 6])


def activation_by_name(inputs, activation="relu", name=None):
    layer_name = name and activation and name + activation
    if activation == "hard_swish":
        return hard_swish(inputs, name=name)
    elif activation.lower() == "prelu":
        shared_axes = list(range(1, len(inputs.shape)))
        shared_axes.pop(-1 if K.image_data_format() == "channels_last" else 0)
        # print(f"{shared_axes = }")
        return keras.layers.PReLU(shared_axes=shared_axes, alpha_initializer=tf.initializers.Constant(0.25), name=layer_name)(inputs)
    elif activation:
        return keras.layers.Activation(activation=activation, name=layer_name)(inputs)
    else:
        return inputs


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, epsilon=BATCH_NORM_EPSILON, name=None):
    """Performs a batch normalization followed by an activation."""
    bn_axis = -1 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=epsilon,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if activation:
        nn = activation_by_name(nn, activation=activation, name=name)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", use_bias=False, groups=1, use_torch_padding=True, name=None, **kwargs):
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if use_torch_padding and padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"
    return keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        groups=groups,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "conv",
        **kwargs,
    )(inputs)


def se_block(inputs, se_ratio=0.25, activation="relu", use_bias=True, name=None):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    reduction = int(filters * se_ratio)
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "1_conv")(se)
    se = activation_by_name(se, activation=activation, name=name)
    se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "2_conv")(se)
    se = activation_by_name(se, activation="sigmoid", name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, se])


def separable_conv2d(inputs, hidden_ratio, out_channel, kernel_size=3, strides=1, group_size=8, activation="relu", name=None):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_filters = inputs.shape[channel_axis]
    hidden_dim = int(input_filters * hidden_ratio)
    nn = conv2d_no_bias(inputs, hidden_dim, kernel_size, strides, padding="SAME", groups=input_filters // group_size, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
    nn = conv2d_no_bias(nn, out_channel, 1, 1, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")
    return nn


def vargface_block(inputs, out_channel, conv_shortcut=False, is_merge=False, strides=1, hidden_ratio=2, se_ratio=0, activation="relu", name=None):
    kernel_size = 3
    if conv_shortcut:
        shortcut = separable_conv2d(inputs, hidden_ratio, out_channel, kernel_size, strides, activation=activation, name=name + "shortcut_")
    else:
        shortcut = inputs

    if is_merge:
        sep_1 = separable_conv2d(inputs, hidden_ratio, out_channel, kernel_size, strides, activation=activation, name=name + "sep1_1_")
        sep_2 = separable_conv2d(inputs, hidden_ratio, out_channel, kernel_size, strides, activation=activation, name=name + "sep1_2_")
        nn = keras.layers.Add(name=name + "merge_")([sep_1, sep_2])
    else:
        nn = separable_conv2d(inputs, hidden_ratio, out_channel, kernel_size, strides, activation=activation, name=name + "sep1_")
    nn = activation_by_name(nn, activation, name=name + "sep1_")
    nn = separable_conv2d(nn, hidden_ratio, out_channel, kernel_size, 1, activation=activation, name=name + "sep2_")

    if se_ratio > 0:
        nn = se_block(nn, se_ratio, activation=activation, name=name + "se_")
    # print(shortcut.shape, nn.shape)
    out = keras.layers.Add()([shortcut, nn])
    out = activation_by_name(out, activation, name=name + "output_")
    return out


def vargface_stack(inputs, num_block, out_channel, strides=2, hidden_ratio=2, se_ratio=0, activation="relu", name=None):
    nn = inputs
    for id in range(num_block):
        block_name = name + "block{}_".format(id + 1)
        is_merge = True if id == 0 else False
        conv_shortcut = True if id == 0 else False
        cur_strides = strides if id == 0 else 1
        cur_se_ratio = 0 if id == 0 else se_ratio
        nn = vargface_block(nn, out_channel, conv_shortcut, is_merge, cur_strides, hidden_ratio, cur_se_ratio, activation, block_name)
    return nn


def vargface_stem(inputs, stem_width=32, strides=1, se_ratio=0, activation="relu", name=None):
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=strides, padding="SAME", name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
    nn = vargface_block(nn, stem_width, conv_shortcut=True, strides=2, hidden_ratio=1, se_ratio=se_ratio, activation=activation, name=name)
    return nn


def vargface_output(inputs, output_num_features, group_size=8, activation="relu", name=None):
    nn = conv2d_no_bias(inputs, output_num_features, 1, 1, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
    nn = conv2d_no_bias(nn, output_num_features, kernel_size=int(nn.shape[1]), groups=output_num_features // group_size, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")
    return nn


def VargFace(
    num_blocks=[3, 7, 4],
    out_channels=[64, 128, 256],
    channel_expand=1,
    hidden_ratio=2,
    stem_width=32,
    stem_strides=1,
    se_ratio=0,
    input_shape=(112, 112, 3),
    activation="relu",
    output_num_features=1024,
    model_name="vargface",
):
    inputs = keras.layers.Input(input_shape)
    stem_width = int(stem_width * channel_expand)
    out_channels = [int(ii * channel_expand) for ii in out_channels]

    nn = vargface_stem(inputs, stem_width, stem_strides, se_ratio=se_ratio, activation=activation, name="stem_")
    for id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
        name = "stack{}_".format(id + 1)
        nn = vargface_stack(nn, num_block, out_channel, hidden_ratio=hidden_ratio, se_ratio=se_ratio, activation=activation, name=name)

    nn = vargface_output(nn, output_num_features, activation=activation, name="output_")
    model = keras.models.Model(inputs, nn, name=model_name)
    return model
