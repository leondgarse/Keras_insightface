import tensorflow as tf
from tensorflow.keras import layers, models, initializers


def se_block(inputs, reduction=16, name=""):
    input_channels = inputs.shape[-1]
    nn = layers.GlobalAveragePooling2D(keepdims=True)(inputs)
    # nn = Reshape((1, 1, input_channels))(nn)
    nn = layers.Conv2D(input_channels // reduction, kernel_size=1, name=name + "1_conv")(nn)
    nn = layers.PReLU(shared_axes=[1, 2], alpha_initializer=initializers.Constant(0.25), name=name + "prelu")(nn)
    nn = layers.Conv2D(input_channels, kernel_size=1, name=name + "2_conv")(nn)
    nn = layers.Activation(activation="sigmoid")(nn)
    nn = layers.Multiply(name=name + "out")([inputs, nn])
    return nn


def conv_bn_prelu(inputs, filters=-1, kernel_size=1, strides=1, padding="SAME", use_depthwise=False, use_separable=False, activation="prelu", name=""):
    filters = filters if filters > 0 else inputs.shape[-1]
    if use_depthwise:
        nn = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False, name=name + "depthwise")(inputs)
    elif use_separable:
        nn = layers.SeparableConv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False, name=name + "separable")(inputs)
        # depthwise = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False, name=name + "depthwise")(inputs)
        # nn = layers.Conv2D(filters, kernel_size=1, strides=1, padding="VALID", use_bias=False, name=name + "pointwise")(depthwise)
    else:
        nn = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name + "conv")(inputs)

    nn = layers.BatchNormalization(name=name + "bn")(nn)

    if activation is not None and activation.lower() == "prelu":
        nn = layers.PReLU(shared_axes=[1, 2], alpha_initializer=initializers.Constant(0.25), name=name + "prelu")(nn)
    elif activation is not None:
        nn = layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def bottleneck(inputs, filters, expand_ratio=1, kernel_size=1, strides=1, use_residual=False, use_se=False, name=""):
    hidden_channels = int(inputs.shape[-1] * expand_ratio)

    nn = conv_bn_prelu(inputs, hidden_channels, name=name + "1_")
    nn = conv_bn_prelu(nn, kernel_size=kernel_size, strides=strides, use_depthwise=True, name=name + "2_")
    nn = conv_bn_prelu(nn, filters, activation=None, name=name + "3_")

    nn = se_block(nn, name=name + "se_") if use_se else nn
    nn = layers.Add()([inputs, nn]) if use_residual else nn
    return nn


def MobileFaceNet(
    num_blocks=[5, 1, 6, 1, 2],
    out_channels=[64, 128, 128, 128, 128],
    strides=[2, 2, 1, 2, 1],
    expand_ratios=[2, 4, 2, 4, 2],
    use_se=False,
    emb_shape=256,
    input_shape=(112, 112, 3),
    dropout=0,
    pretrained=None,
    include_top=False,
    name="mobile_facenet",
):
    inputs = layers.Input(shape=input_shape)  # (112, 112, 3)
    nn = conv_bn_prelu(inputs, filters=64, kernel_size=3, strides=2, name="stem_1_")  # (56, 56, 64)
    nn = conv_bn_prelu(nn, filters=64, kernel_size=3, strides=1, use_separable=True, name="stem_2_")  # (56, 56, 64)

    for id, (num_block, out_channel, stride, expand_ratio) in enumerate(zip(num_blocks, out_channels, strides, expand_ratios)):
        stack_name = "stack{}_".format(id + 1)
        for block_id in range(num_block):
            cur_strides = stride if block_id == 0 else 1
            use_residual = False if block_id == 0 else True
            block_name = stack_name + "block{}_".format(block_id + 1)
            nn = bottleneck(nn, out_channel, expand_ratio, kernel_size=3, strides=cur_strides, use_residual=use_residual, use_se=use_se, name=block_name)

    if include_top:
        """pointwise_conv"""
        nn = conv_bn_prelu(nn, filters=512, name="header_pointwise_")

        """ GDC """
        nn = layers.DepthwiseConv2D(nn.shape[1], use_bias=False, name="header_gdc_depthwise")(nn)
        nn = layers.BatchNormalization(name="header_gdc_bn")(nn)

        if dropout > 0 and dropout < 1:
            nn = layers.Dropout(dropout)(nn)
        nn = layers.Conv2D(emb_shape, 1, use_bias=False, name="header_gdc_post_conv")(nn)
        nn = layers.Flatten()(nn)
        nn = layers.BatchNormalization(name="pre_embedding")(nn)
        nn = layers.Activation("linear", dtype="float32", name="embedding")(nn)

    model = models.Model(inputs=inputs, outputs=nn, name=name)
    if pretrained:
        model.load_weights(pretrained)
    return model
