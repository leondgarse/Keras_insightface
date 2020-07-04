from tensorflow.keras import backend as K
import tensorflow as tf


def bottleneck(
    inputs, filters, expansion_filters, kernel_size, alpha=1.0, strides=(1, 1), use_se=False, activation=tf.nn.relu6
):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    in_channels = inputs.shape[channel_axis]
    filters = _make_divisible(filters * alpha, 8)
    xx = tf.keras.layers.Conv2D(
        expansion_filters, 1, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(inputs)
    xx = tf.keras.layers.BatchNormalization()(xx)
    xx = activation(xx)
    if strides == 2:
        xx = tf.keras.layers.ZeroPadding2D(((kernel_size - 1) // 2, (kernel_size - 1) // 2))(xx)
    xx = tf.keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        use_bias=False,
        padding="same" if strides == 1 else "valid",
        depthwise_regularizer=tf.keras.regularizers.l2(1e-5),
    )(xx)
    xx = tf.keras.layers.BatchNormalization()(xx)
    if use_se:
        xx = se_block(xx)
    xx = activation(xx)
    xx = tf.keras.layers.Conv2D(
        filters, kernel_size=1, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(xx)
    xx = tf.keras.layers.BatchNormalization()(xx)
    if in_channels == filters and strides == 1:
        xx = tf.keras.layers.Add()([inputs, xx])
    return xx


def se_block(inputs, reduction=4):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs.shape[channel_axis]
    se = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    se = tf.keras.layers.Dense(filters // reduction, activation="relu", use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation="sigmoid", use_bias=False)(se)
    # if K.image_data_format() == 'channels_first':
    #     se = Permute((3, 1, 2))(se)
    x = tf.keras.layers.Multiply()([inputs, se])
    return x


def h_swish(inputs):
    return inputs * tf.nn.relu6(inputs + 3) / 6


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobilenetV3(input_shape, num_classes=0, size="large", include_top=True, alpha=1.0):
    input = tf.keras.layers.Input(input_shape)
    first_block_filters = _make_divisible(16 * alpha, 8)
    if size not in ["large", "small"]:
        raise ValueError("size should be large or small")
    if size == "large":
        xx = tf.keras.layers.Conv2D(
            first_block_filters, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(input)
        xx = tf.keras.layers.BatchNormalization()(xx)
        xx = h_swish(xx)
        xx = bottleneck(xx, 16, 16, 3, alpha=alpha, strides=1, use_se=False, activation=tf.nn.relu6)
        xx = bottleneck(xx, 24, 64, 3, alpha=alpha, strides=2, use_se=False, activation=tf.nn.relu6)
        xx = bottleneck(xx, 24, 72, 3, alpha=alpha, strides=1, use_se=False, activation=tf.nn.relu6)
        xx = bottleneck(xx, 40, 72, 5, alpha=alpha, strides=2, use_se=True, activation=tf.nn.relu6)
        xx = bottleneck(xx, 40, 120, 5, alpha=alpha, strides=1, use_se=True, activation=tf.nn.relu6)
        xx = bottleneck(xx, 40, 120, 5, alpha=alpha, strides=1, use_se=True, activation=tf.nn.relu6)
        xx = bottleneck(xx, 80, 240, 3, alpha=alpha, strides=2, use_se=False, activation=h_swish)
        xx = bottleneck(xx, 80, 200, 3, alpha=alpha, strides=1, use_se=False, activation=h_swish)
        xx = bottleneck(xx, 80, 184, 3, alpha=alpha, strides=1, use_se=False, activation=h_swish)
        xx = bottleneck(xx, 80, 184, 3, alpha=alpha, strides=1, use_se=False, activation=h_swish)
        xx = bottleneck(xx, 112, 480, 3, alpha=alpha, strides=1, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 112, 672, 3, alpha=alpha, strides=1, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 160, 672, 5, alpha=alpha, strides=2, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 160, 960, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 160, 960, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)
        xx = tf.keras.layers.Conv2D(
            _make_divisible(960 * alpha, 8), 1, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(xx)
        xx = tf.keras.layers.BatchNormalization()(xx)
        output = h_swish(xx)
        name = "MobilenetV3_large"
    else:
        xx = tf.keras.layers.Conv2D(
            first_block_filters, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(input)
        xx = tf.keras.layers.BatchNormalization()(xx)
        xx = h_swish(xx)
        xx = bottleneck(xx, 16, 16, 3, alpha=alpha, strides=2, use_se=True, activation=tf.nn.relu6)
        xx = bottleneck(xx, 24, 72, 3, alpha=alpha, strides=2, use_se=False, activation=tf.nn.relu6)
        xx = bottleneck(xx, 24, 88, 3, alpha=alpha, strides=1, use_se=False, activation=tf.nn.relu6)
        xx = bottleneck(xx, 40, 96, 5, alpha=alpha, strides=2, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 40, 240, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 40, 240, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 48, 120, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 48, 144, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 96, 288, 5, alpha=alpha, strides=2, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 96, 576, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)
        xx = bottleneck(xx, 96, 576, 5, alpha=alpha, strides=1, use_se=True, activation=h_swish)
        xx = tf.keras.layers.Conv2D(
            _make_divisible(576 * alpha, 8), 1, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(xx)
        xx = tf.keras.layers.BatchNormalization()(xx)
        xx = se_block(xx)
        output = h_swish(xx)
        name = "MobilenetV3_small"
    if include_top:
        output = tf.keras.layers.AveragePooling2D(pool_size=x.shape[1:3])(output)
        if alpha > 1.0:
            last_block_filters = _make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280
        output = tf.keras.layers.Conv2D(
            last_block_filters, 1, use_bias=False, activation=h_swish, kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(output)
        output = tf.keras.layers.Dropout(0.8)(output)
        output = tf.keras.layers.Conv2D(
            num_classes,
            1,
            use_bias=True,
            activation=tf.keras.activations.softmax,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        )(output)
        output = tf.keras.layers.Flatten()(output)
    return tf.keras.Model(input, output, name=name)
