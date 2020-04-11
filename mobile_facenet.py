from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    PReLU,
    SeparableConv2D,
    DepthwiseConv2D,
    add,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model

"""Building Block Functions"""


def conv_block(inputs, filters, kernel_size, strides, padding):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    Z = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
    Z = BatchNormalization(axis=channel_axis)(Z)
    A = PReLU(shared_axes=[1, 2])(Z)
    return A


def separable_conv_block(inputs, filters, kernel_size, strides):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    Z = SeparableConv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(inputs)
    Z = BatchNormalization(axis=channel_axis)(Z)
    A = PReLU(shared_axes=[1, 2])(Z)
    return A


def bottleneck(inputs, filters, kernel, t, s, r=False):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    Z1 = conv_block(inputs, tchannel, 1, 1, "same")
    Z1 = DepthwiseConv2D(kernel, strides=s, padding="same", depth_multiplier=1, use_bias=False)(Z1)
    Z1 = BatchNormalization(axis=channel_axis)(Z1)
    A1 = PReLU(shared_axes=[1, 2])(Z1)
    Z2 = Conv2D(filters, 1, strides=1, padding="same", use_bias=False)(A1)
    Z2 = BatchNormalization(axis=channel_axis)(Z2)
    if r:
        Z2 = add([Z2, inputs])
    return Z2


def inverted_residual_block(inputs, filters, kernel, t, strides, n):
    Z = bottleneck(inputs, filters, kernel, t, strides)
    for i in range(1, n):
        Z = bottleneck(Z, filters, kernel, t, 1, True)
    return Z


def linear_GD_conv_block(inputs, kernel_size, strides):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    Z = DepthwiseConv2D(kernel_size, strides=strides, padding="valid", depth_multiplier=1, use_bias=False)(inputs)
    Z = BatchNormalization(axis=channel_axis)(Z)
    return Z


def mobile_facenet(emb_shape=128, dropout=1, name="mobile_facenet", weight_file=None):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    if K.image_data_format() == "channels_first":
        X = Input(shape=(3, 112, 112))
    else:
        X = Input(shape=(112, 112, 3))
    M = conv_block(X, 64, 3, 2, "same")  # Output Shape: (56, 56, 64)
    M = separable_conv_block(M, 64, 3, 1)  # (56, 56, 64)
    M = inverted_residual_block(M, 64, 3, t=2, strides=2, n=5)  # (28, 28, 64)
    M = inverted_residual_block(M, 128, 3, t=4, strides=2, n=1)  # (14, 14, 128)
    M = inverted_residual_block(M, 128, 3, t=2, strides=1, n=6)  # (14, 14, 128)
    M = inverted_residual_block(M, 128, 3, t=4, strides=2, n=1)  # (7, 7, 128)
    M = inverted_residual_block(M, 128, 3, t=2, strides=1, n=2)  # (7, 7, 128)
    M = conv_block(M, 512, 1, 1, "valid")  # (7, 7, 512)
    M = linear_GD_conv_block(M, 7, 1)  # (1, 1, 512)
    # kernel_size = 7 for 112 x 112; 4 for 64 x 64

    M = conv_block(M, emb_shape, 1, 1, "valid")
    if dropout > 0 and dropout < 1:
        M = Dropout(rate=dropout)(M)
    M = Flatten()(M)
    M = Dense(emb_shape, activation=None, use_bias=False, kernel_initializer="glorot_normal")(M)
    M = BatchNormalization(axis=channel_axis, name="embedding")(M)

    model = Model(inputs=X, outputs=M, name=name)
    if weight_file:
        model.load_weights(weight_file)
    return model
