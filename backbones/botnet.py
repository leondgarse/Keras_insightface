"""
A Keras version of `botnet`.
Original TensorFlow version: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
"""

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util.tf_export import keras_export

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


@keras_export("keras.layers.MHSAWithRelativePosition")
class MHSAWithRelativePosition(keras.layers.MultiHeadAttention):
    def __init__(self, num_heads=4, bottleneck_dimension=512, relative=True, **kwargs):
        self.key_dim = bottleneck_dimension // num_heads
        super(MHSAWithRelativePosition, self).__init__(num_heads=num_heads, key_dim=self.key_dim, **kwargs)
        self.num_heads, self.bottleneck_dimension, self.relative = num_heads, bottleneck_dimension, relative

    def _build_from_signature(self, query, value, key=None):
        super(MHSAWithRelativePosition, self)._build_from_signature(query=query, value=value)
        if hasattr(query, "shape"):
            _, hh, ww, _ = query.shape
        else:
            _, hh, ww, _ = query
        stddev = self.key_dim ** -0.5
        self.rel_emb_w = self.add_weight(
            name="r_width",
            shape=(self.key_dim, 2 * ww - 1),
            initializer=tf.random_normal_initializer(stddev=stddev),
            trainable=True,
        )
        self.rel_emb_h = self.add_weight(
            name="r_height",
            shape=(self.key_dim, 2 * hh - 1),
            initializer=tf.random_normal_initializer(stddev=stddev),
            trainable=True,
        )

    def get_config(self):
        base_config = super(MHSAWithRelativePosition, self).get_config()
        base_config.pop("key_dim", None)
        base_config.update(
            {"num_heads": self.num_heads, "bottleneck_dimension": self.bottleneck_dimension, "relative": self.relative}
        )
        return base_config

    def rel_to_abs(self, rel_pos):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, height, width, 2*width - 1]
        Output: [bs, heads, height, width, width]
        """
        _, heads, hh, ww, dim = rel_pos.shape
        col_pad = tf.zeros_like(rel_pos[:, :, :, :, :1], dtype=rel_pos.dtype)
        rel_pos = tf.concat([rel_pos, col_pad], axis=-1)
        flat_x = tf.reshape(rel_pos, [-1, heads, hh, ww * 2 * ww])
        flat_pad = tf.zeros_like(flat_x[:, :, :, : ww - 1], dtype=rel_pos.dtype)
        flat_x_padded = tf.concat([flat_x, flat_pad], axis=-1)
        final_x = tf.reshape(flat_x_padded, [-1, heads, hh, ww + 1, 2 * ww - 1])
        final_x = final_x[:, :, :, :ww, ww - 1 :]
        return final_x

    def relative_logits_1d(self, query, rel_k, transpose_mask):
        """
        Compute relative logits along one dimenion.
        `q`: [bs, heads, height, width, dim]
        `rel_k`: [dim, 2*width - 1]
        """
        _, _, hh, _, _ = query.shape
        rel_logits = tf.matmul(query, rel_k)
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = tf.expand_dims(rel_logits, axis=3)
        rel_logits = tf.tile(rel_logits, [1, 1, 1, hh, 1, 1])
        rel_logits = tf.transpose(rel_logits, transpose_mask)
        return rel_logits

    def relative_logits(self, query):
        query = tf.transpose(query, [0, 3, 1, 2, 4])
        rel_logits_w = self.relative_logits_1d(query=query, rel_k=self.rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        query = tf.transpose(query, [0, 1, 3, 2, 4])
        rel_logits_h = self.relative_logits_1d(query=query, rel_k=self.rel_emb_h, transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def call(self, inputs, attention_mask=None, return_attention_scores=False, training=None):
        if not self._built_from_signature:
            self._build_from_signature(query=inputs, value=inputs)
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(inputs)

        # `key` = [B, S, N, H]
        key = self._key_dense(inputs)

        # `value` = [B, S, N, H]
        value = self._value_dense(inputs)

        query = math_ops.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))
        attention_scores = special_math_ops.einsum(self._dot_product_equation, key, query)
        if self.relative:
            attention_scores += self.relative_logits(query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = special_math_ops.einsum(self._combine_equation, attention_scores_dropout, value)

        # attention_output = self._output_dense(attention_output)
        hh, ww = inputs.shape[1], inputs.shape[2]
        attention_output = tf.reshape(attention_output, [-1, hh, ww, self.num_heads * self.key_dim])

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
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
    return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name + "conv")(inputs)


def bot_block(
    featuremap,
    heads=4,
    proj_factor=4,
    activation="relu",
    pos_enc_type="relative",
    strides=1,
    target_dimension=2048,
    name="all2all",
    use_MHSA=True,
):
    if strides != 1 or featuremap.shape[-1] != target_dimension:
        padding = "SAME" if strides == 1 else "VALID"
        shortcut = conv2d_no_bias(featuremap, target_dimension, 1, strides=strides, padding=padding, name=name + "_0_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "_0_")
    else:
        shortcut = featuremap

    bottleneck_dimension = target_dimension // proj_factor

    if use_MHSA:    # BotNet block
        nn = conv2d_no_bias(featuremap, bottleneck_dimension, 1, strides=1, padding="VALID", name=name + "_1_")
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_1_")
        nn = MHSAWithRelativePosition(num_heads=heads, bottleneck_dimension=bottleneck_dimension, name=name + "_2_mhsa")(nn)
        if strides != 1:
            nn = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(nn)
    else:   # ResNet block
        nn = conv2d_no_bias(featuremap, bottleneck_dimension, 1, strides=strides, padding="VALID", name=name + "_1_")
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_1_")
        nn = conv2d_no_bias(nn, bottleneck_dimension, 3, strides=1, padding="SAME", name=name + "_2_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_2_")

    nn = conv2d_no_bias(nn, target_dimension, 1, strides=1, padding="VALID", name=name + "_3_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "_3_")

    nn = layers.Add(name=name + "_add")([shortcut, nn])
    return layers.Activation(activation, name=name + "_out")(nn)


def bot_stack(
    featuremap,
    target_dimension=2048,
    num_layers=3,
    strides=2,
    activation="relu",
    heads=4,
    proj_factor=4,
    pos_enc_type="relative",
    name="all2all_stack",
    use_MHSA=True,
):
    """ c5 Blockgroup of BoT Blocks. Use `activation=swish` for `silu` """
    for i in range(num_layers):
        featuremap = bot_block(
            featuremap,
            heads=heads,
            proj_factor=proj_factor,
            activation=activation,
            pos_enc_type=pos_enc_type,
            strides=strides if i == 0 else 1,
            target_dimension=target_dimension,
            name=name + "_block{}".format(i+1),
            use_MHSA=use_MHSA,
        )
    return featuremap


def bot_block_2(
    featuremap,
    heads=4,
    proj_factor=4,
    activation="relu",
    pos_enc_type="relative",
    strides=1,
    target_dimension=2048,
    conv_shortcut=False,
    use_MHSA=True,
    name="all2all",
):
    preact = batchnorm_with_activation(featuremap, activation=activation, name=name + "_preact_")

    if conv_shortcut:
        shortcut = conv2d_no_bias(preact, target_dimension, 1, strides=strides, name=name + "_0_")
    else:
        shortcut = layers.MaxPooling2D(1, strides=strides)(featuremap) if strides > 1 else featuremap

    bottleneck_dimension = target_dimension // proj_factor
    nn = conv2d_no_bias(preact, bottleneck_dimension, 1, strides=1, name=name + "_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "_1_")

    if use_MHSA:    # BotNet block
        nn = MHSAWithRelativePosition(num_heads=heads, bottleneck_dimension=bottleneck_dimension, name=name + "_2_mhsa")(nn)
        if strides != 1:
            nn = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name=name + '_2_pool')(nn)
    else:   # ResNet block
        nn = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(nn)
        nn = conv2d_no_bias(nn, bottleneck_dimension, 3, strides=strides, name=name + "_2_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "_2_")

    nn = conv2d_no_bias(nn, target_dimension, 1, strides=1, name=name + "_3_")
    nn = layers.Add(name=name + '_out')([shortcut, nn])
    return nn


def bot_stack_2(
    featuremap,
    target_dimension=2048,
    num_layers=3,
    strides=2,
    activation="relu",
    heads=4,
    proj_factor=4,
    pos_enc_type="relative",
    name="all2all_stack",
    use_MHSA=True,
):
    """ c5 Blockgroup of BoT Blocks. Use `activation=swish` for `silu` """
    for ii in range(num_layers):
        featuremap = bot_block_2(
            featuremap,
            heads=heads,
            proj_factor=proj_factor,
            activation=activation,
            pos_enc_type=pos_enc_type,
            strides=strides if ii == num_layers - 1 else 1,
            target_dimension=target_dimension,
            conv_shortcut=True if ii == 0 else False,
            use_MHSA=use_MHSA,
            name=name + "_block{}".format(ii+1),
        )
    return featuremap


def BotNet(
    stack_fn,
    preact,
    use_bias,
    model_name="botnet",
    activation="relu",
    include_top=True,
    weights=None,
    input_shape=None,
    classes=1000,
    use_conv3_stride1=False,
    classifier_activation="softmax",
):
    img_input = layers.Input(shape=input_shape)

    if use_conv3_stride1:
        nn = layers.Conv2D(64, 3, strides=1, use_bias=use_bias, padding="SAME", name="conv1_conv")(img_input)
    else:
        nn = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
        nn = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(nn)

    if not preact:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="conv1_")
    nn = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(nn)
    nn = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(nn)

    nn = stack_fn(nn)
    if preact:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="post_")
    if include_top:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(classes, activation=classifier_activation, name="predictions")(nn)
    return keras.models.Model(img_input, nn, name=model_name)


def BotNet50(strides=2, activation="relu", include_top=True, weights=None, input_shape=None, classes=1000, use_MHSA=True, **kwargs):
    def stack_fn(nn):
        nn = bot_stack(nn, 64 * 4, 3, strides=1, activation=activation, name="conv2", use_MHSA=False)
        nn = bot_stack(nn, 128 * 4, 4, strides=2, activation=activation, name="conv3", use_MHSA=False)
        nn = bot_stack(nn, 256 * 4, 6, strides=2, activation=activation, name="conv4", use_MHSA=False)
        nn = bot_stack(nn, 512 * 4, 3, strides=strides, activation=activation, use_MHSA=use_MHSA)
        return nn

    return BotNet(stack_fn, False, True, "botnet50", activation, include_top, weights, input_shape, classes, **kwargs)


def BotNet101(strides=2, activation="relu", include_top=True, weights=None, input_shape=None, classes=1000, use_MHSA=True, **kwargs):
    def stack_fn(nn):
        nn = bot_stack(nn, 64 * 4, 3, strides=1, activation=activation, name="conv2", use_MHSA=False)
        nn = bot_stack(nn, 128 * 4, 4, strides=2, activation=activation, name="conv3", use_MHSA=False)
        nn = bot_stack(nn, 256 * 4, 23, strides=2, activation=activation, name="conv4", use_MHSA=False)
        nn = bot_stack(nn, 512 * 4, 3, strides=strides, activation=activation, use_MHSA=use_MHSA)
        return nn

    return BotNet(stack_fn, False, True, "botnet50", activation, include_top, weights, input_shape, classes, **kwargs)


def BotNet152(strides=2, activation="relu", include_top=True, weights=None, input_shape=None, classes=1000, use_MHSA=True, **kwargs):
    def stack_fn(nn):
        nn = bot_stack(nn, 64 * 4, 3, strides=1, activation=activation, name="conv2", use_MHSA=False)
        nn = bot_stack(nn, 128 * 4, 8, strides=2, activation=activation, name="conv3", use_MHSA=False)
        nn = bot_stack(nn, 256 * 4, 36, strides=2, activation=activation, name="conv4", use_MHSA=False)
        nn = bot_stack(nn, 512 * 4, 3, strides=strides, activation=activation, use_MHSA=use_MHSA)
        return nn

    return BotNet(stack_fn, False, True, "botnet50", activation, include_top, weights, input_shape, classes, **kwargs)


def BotNet50V2(strides=2, activation="relu", include_top=True, weights=None, input_shape=None, classes=1000, use_MHSA=True, **kwargs):
    def stack_fn(nn):
        nn = bot_stack_2(nn, 64 * 4, 3, name='conv2', use_MHSA=False)
        nn = bot_stack_2(nn, 128 * 4, 4, name='conv3', use_MHSA=False)
        nn = bot_stack_2(nn, 256 * 4, 6, strides=strides, name='conv4', use_MHSA=False)
        return bot_stack_2(nn, 512 * 4, 3, strides=1, name='conv5', use_MHSA=use_MHSA)

    return BotNet(stack_fn, True, True, "botnet50v2", activation, include_top, weights, input_shape, classes, **kwargs)

def BotNet101V2(strides=2, activation="relu", include_top=True, weights=None, input_shape=None, classes=1000, use_MHSA=True, **kwargs):
    def stack_fn(nn):
        nn = bot_stack_2(nn, 64 * 4, 3, name='conv2', use_MHSA=False)
        nn = bot_stack_2(nn, 128 * 4, 4, name='conv3', use_MHSA=False)
        nn = bot_stack_2(nn, 256 * 4, 23, strides=strides, name='conv4', use_MHSA=False)
        return bot_stack_2(nn, 512 * 4, 3, strides=1, name='conv5', use_MHSA=use_MHSA)

    return BotNet(stack_fn, True, True, "botnet101v2", activation, include_top, weights, input_shape, classes, **kwargs)
