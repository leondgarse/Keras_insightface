# Copied from https://github.com/QiaoranC/tf_ResNeSt_RegNet_model, and modified

import tensorflow as tf
from tensorflow.keras.activations import softmax
from tensorflow.keras import models
from tensorflow.keras.layers import (
    AveragePooling2D,
    Activation,
    Input,
    Conv2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    MaxPool2D,
    Add,
)


class GroupedConv2D(object):
    """Groupped convolution.
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    Currently tf.keras and tf.layers don't support group convolution, so here we
    use split/concat to implement this op. It reuses kernel_size for group
    definition, where len(kernel_size) is number of groups. Notably, it allows
    different group has different kernel size.
    """

    def __init__(self, filters, kernel_size, **kwargs):
        """Initialize the layer.
        Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original Conv2D. If it is a list, then we split the channels
            and perform different kernel for each group.
        **kwargs: other parameters passed to the original conv2d layer.
        """
        self._groups = len(kernel_size)
        self._channel_axis = -1

        self._convs = []
        splits = self._split_channels(filters, self._groups)
        for i in range(self._groups):
            self._convs.append(self._get_conv2d(splits[i], kernel_size[i], **kwargs))

    def _get_conv2d(self, filters, kernel_size, **kwargs):
        """A helper function to create Conv2D layer."""
        return Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def __call__(self, inputs):
        if len(self._convs) == 1:
            return self._convs[0](inputs)

        if tf.__version__ < "2.0.0":
            filters = inputs.shape[self._channel_axis].value
        else:
            filters = inputs.shape[self._channel_axis]
        splits = self._split_channels(filters, len(self._convs))
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
        x = tf.concat(x_outputs, self._channel_axis)
        return x


class ResNest:
    def __init__(
        self,
        verbose=False,
        input_shape=(224, 224, 3),
        active="relu",
        blocks_set=[3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        block_expansion=4,
        avg_down=True,
        avd=True,
        avd_first=False,
        preact=False,
        using_basic_block=False,
        name="model",
    ):
        self.channel_axis = -1  # not for change
        self.verbose = verbose
        self.active = active  # default relu
        self.input_shape = input_shape
        # self.n_classes = n_classes
        # self.fc_activation = fc_activation
        self.name = name

        self.blocks_set = blocks_set
        self.radix = radix
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.block_expansion = block_expansion
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first

        # self.cardinality = 1
        self.preact = preact
        self.using_basic_block = using_basic_block

    def _make_stem(self, input_tensor, stem_width=64, deep_stem=False):
        x = Conv2D(stem_width, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal", use_bias=False)(
            input_tensor
        )
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(stem_width, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(stem_width * 2, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False)(x)

        return x

    def _rsoftmax(self, input_tensor, filters, radix, groups):
        x = input_tensor
        batch = x.shape[0]
        x = tf.reshape(x, [-1, groups, radix, filters // groups])
        x = tf.transpose(x, [0, 2, 1, 3])
        x = softmax(x, axis=1)
        x = tf.reshape(x, [-1, 1, 1, radix * filters])
        return x

    def _SplAtConv2d(self, input_tensor, filters=64, kernel_size=3, stride=1, groups=1, radix=0):
        x = input_tensor
        in_channels = input_tensor.shape[-1]

        x = GroupedConv2D(
            filters=filters * radix,
            kernel_size=[kernel_size for i in range(groups * radix)],
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )(x)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        batch, rchannel = x.shape[0], x.shape[-1]
        splited = tf.split(x, radix, axis=-1)
        gap = sum(splited)

        # print('sum',gap.shape)
        gap = GlobalAveragePooling2D()(gap)
        gap = tf.reshape(gap, [-1, 1, 1, filters])
        # print('adaptive_avg_pool2d',gap.shape)

        reduction_factor = 4
        inter_channels = max(in_channels * radix // reduction_factor, 32)

        x = Conv2D(inter_channels, kernel_size=1)(gap)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(filters * radix, kernel_size=1)(x)

        atten = self._rsoftmax(x, filters, radix, groups)

        logits = tf.split(atten, radix, axis=-1)
        out = sum([a * b for a, b in zip(splited, logits)])
        return out

    def _make_block(self, input_tensor, filters=64, stride=2, radix=1):
        x = input_tensor
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * 4:
            short_cut = input_tensor
            short_cut = AveragePooling2D(pool_size=stride, strides=stride, padding="same")(short_cut)
            short_cut = Conv2D(
                filters * 4, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False
            )(short_cut)
            short_cut = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(short_cut)
        else:
            short_cut = input_tensor

        group_width = filters
        x = Conv2D(group_width, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        x = self._SplAtConv2d(x, filters=group_width, kernel_size=3, stride=1, groups=1, radix=radix)
        if stride > 1:
            x = AveragePooling2D(pool_size=3, strides=stride, padding="same")(x)
            # print('can')
        x = Conv2D(filters * 4, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)

        m2 = Add()([x, short_cut])
        m2 = Activation(self.active)(m2)
        return m2

    def _make_layer(self, input_tensor, blocks=4, filters=64, stride=2):
        x = self._make_block(input_tensor, filters=filters, stride=stride, radix=2)
        for i in range(1, blocks):
            x = self._make_block(x, filters=filters, stride=1, radix=2)
        return x

    def build(self):
        input_sig = Input(shape=self.input_shape)
        x = self._make_stem(input_sig, stem_width=self.stem_width, deep_stem=self.deep_stem)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = MaxPool2D(pool_size=3, strides=2, padding="same")(x)

        x = self._make_layer(x, blocks=self.blocks_set[0], filters=64, stride=1)
        x = self._make_layer(x, blocks=self.blocks_set[1], filters=128, stride=2)
        x = self._make_layer(x, blocks=self.blocks_set[2], filters=256, stride=2)
        x = self._make_layer(x, blocks=self.blocks_set[3], filters=512, stride=2)

        model = models.Model(inputs=input_sig, outputs=x, name=self.name)

        return model


def ResNest50(input_shape=(112, 112, 3), verbose=False, model_name="ResNest50", **kwargs):
    return ResNest(
        verbose=verbose, input_shape=input_shape, blocks_set=[3, 4, 6, 3], stem_width=32, name=model_name, **kwargs
    ).build()


def ResNest101(input_shape=(112, 112, 3), verbose=False, model_name="ResNest101", **kwargs):
    return ResNest(
        verbose=verbose, input_shape=input_shape, blocks_set=[3, 4, 23, 3], stem_width=64, name=model_name, **kwargs
    ).build()
