"""
A Keras version of `botnet`.
Original TensorFlow version: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


@tf.keras.utils.register_keras_serializable(package="Custom")
class MHSAWithPositionEmbedding(keras.layers.MultiHeadAttention):
    def __init__(self, num_heads=4, bottleneck_dimension=512, relative=True, **kwargs):
        self.key_dim = bottleneck_dimension // num_heads
        super(MHSAWithPositionEmbedding, self).__init__(num_heads=num_heads, key_dim=self.key_dim, **kwargs)
        self.num_heads, self.bottleneck_dimension, self.relative = num_heads, bottleneck_dimension, relative

    def _build_from_signature(self, query, value, key=None):
        super(MHSAWithPositionEmbedding, self)._build_from_signature(query=query, value=value)
        if hasattr(query, "shape"):
            _, hh, ww, _ = query.shape
        else:
            _, hh, ww, _ = query
        stddev = self.key_dim ** -0.5

        if self.relative:
            # Relative positional embedding
            pos_emb_w_shape = (self.key_dim, 2 * ww - 1)
            pos_emb_h_shape = (self.key_dim, 2 * hh - 1)
        else:
            # Absolute positional embedding
            pos_emb_w_shape = (self.key_dim, ww)
            pos_emb_h_shape = (self.key_dim, hh)

        self.pos_emb_w = self.add_weight(
            name="r_width",
            shape=pos_emb_w_shape,
            initializer=tf.random_normal_initializer(stddev=stddev),
            trainable=True,
        )
        self.pos_emb_h = self.add_weight(
            name="r_height",
            shape=pos_emb_h_shape,
            initializer=tf.random_normal_initializer(stddev=stddev),
            trainable=True,
        )

    def get_config(self):
        base_config = super(MHSAWithPositionEmbedding, self).get_config()
        base_config.pop("key_dim", None)
        base_config.update({"num_heads": self.num_heads, "bottleneck_dimension": self.bottleneck_dimension, "relative": self.relative})
        return base_config

    def rel_to_abs(self, rel_pos):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, height, width, 2*width - 1]
        Output: [bs, heads, height, width, width]
        """
        _, heads, hh, ww, dim = rel_pos.shape  # [bs, heads, height, width, 2 * width - 1]
        # [bs, heads, height, width * (2 * width - 1)] --> [bs, heads, height, width * (2 * width - 1) - width]
        flat_x = tf.reshape(rel_pos, [-1, heads, hh, ww * (ww * 2 - 1)])[:, :, :, ww - 1 : -1]
        # [bs, heads, height, width, 2 * (width - 1)] --> [bs, heads, height, width, width]
        return tf.reshape(flat_x, [-1, heads, hh, ww, 2 * (ww - 1)])[:, :, :, :, :ww]

    def relative_logits(self, query):
        query_w = tf.transpose(query, [0, 3, 1, 2, 4])  # e.g.: [1, 4, 14, 16, 128], [bs, heads, hh, ww, dims]
        rel_logits_w = tf.matmul(query_w, self.pos_emb_w)   # [1, 4, 14, 16, 31], 2 * 16 - 1 == 31
        rel_logits_w = self.rel_to_abs(rel_logits_w)    # [1, 4, 14, 16, 16]

        query_h = tf.transpose(query, [0, 3, 2, 1, 4]) # [1, 4, 16, 14, 128], [bs, heads, ww, hh, dims], Exchange `ww` and `hh`
        rel_logits_h = tf.matmul(query_h, self.pos_emb_h)  # [1, 4, 16, 14, 27], 2 * 14 - 1 == 27
        rel_logits_h = self.rel_to_abs(rel_logits_h)  # [1, 4, 16, 14, 14]
        rel_logits_h = tf.transpose(rel_logits_h, [0, 1, 3, 2, 4]) # [1, 4, 14, 16, 14], transpose back

        return tf.expand_dims(rel_logits_w, axis=-2) + tf.expand_dims(rel_logits_h, axis=-1) # [1, 4, 14, 16, 14, 16]

    def absolute_logits(self, query):
        pos_emb = tf.expand_dims(self.pos_emb_h, 2) + tf.expand_dims(self.pos_emb_w, 1)
        abs_logits = tf.einsum("bxyhd,dpq->bhxypq", query, pos_emb)
        return abs_logits

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

        query = tf.multiply(query, 1.0 / tf.math.sqrt(float(self._key_dim)))
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        if self.relative:
            # Add relative positional embedding
            attention_scores += self.relative_logits(query)
        else:
            # Add absolute positional embedding
            attention_scores += self.absolute_logits(query)

        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)

        # attention_output = self._output_dense(attention_output)
        hh, ww = inputs.shape[1], inputs.shape[2]
        attention_output = tf.reshape(attention_output, [-1, hh, ww, self.num_heads * self.key_dim])

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def load_resized_pos_emb(self, source_layer):
        # For input 224 --> [128, 27], convert to 480 --> [128, 30]
        image_like_w = tf.expand_dims(tf.transpose(source_layer.pos_emb_w, [1, 0]), 0)
        resize_w = tf.image.resize(image_like_w, (1, self.pos_emb_w.shape[1]))[0]
        self.pos_emb_w.assign(tf.transpose(resize_w, [1, 0]))

        image_like_h = tf.expand_dims(tf.transpose(source_layer.pos_emb_h, [1, 0]), 0)
        resize_h = tf.image.resize(image_like_h, (1, self.pos_emb_h.shape[1]))[0]
        self.pos_emb_h.assign(tf.transpose(resize_h, [1, 0]))


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
    if padding == "SAME":
        inputs = layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(inputs)
    return layers.Conv2D(filters, kernel_size, strides=strides, padding="VALID", use_bias=False, name=name + "conv")(inputs)


def bot_block(
    featuremap,
    heads=4,
    proj_factor=4,
    activation="relu",
    relative_pe=True,
    strides=1,
    target_dimension=2048,
    name="all2all",
    use_MHSA=True,
):
    if strides != 1 or featuremap.shape[-1] != target_dimension:
        # padding = "SAME" if strides == 1 else "VALID"
        shortcut = conv2d_no_bias(featuremap, target_dimension, 1, strides=strides, name=name + "_shorcut_")
        bn_act = activation if use_MHSA else None
        shortcut = batchnorm_with_activation(shortcut, activation=bn_act, zero_gamma=False, name=name + "_shorcut_")
    else:
        shortcut = featuremap

    bottleneck_dimension = target_dimension // proj_factor

    nn = conv2d_no_bias(featuremap, bottleneck_dimension, 1, strides=1, padding="VALID", name=name + "_1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_1_")

    if use_MHSA:  # BotNet block
        nn = MHSAWithPositionEmbedding(num_heads=heads, bottleneck_dimension=bottleneck_dimension, relative=relative_pe, use_bias=False, name=name + "_2_mhsa")(
            nn
        )
        if strides != 1:
            nn = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(nn)
    else:  # ResNet block
        nn = conv2d_no_bias(nn, bottleneck_dimension, 3, strides=strides, padding="SAME", name=name + "_2_")
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
    relative_pe=True,
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
            relative_pe=relative_pe,
            strides=strides if i == 0 else 1,
            target_dimension=target_dimension,
            name=name + "block{}".format(i + 1),
            use_MHSA=use_MHSA,
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
    classifier_activation="softmax",
    **kwargs
):
    img_input = layers.Input(shape=input_shape)

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


def BotNet50(strides=2, activation="relu", relative_pe=True, include_top=True, weights=None, input_tensor=None, input_shape=None, classes=1000, **kwargs):
    def stack_fn(nn):
        nn = bot_stack(nn, 64 * 4, 3, strides=1, activation=activation, name="stack1_", use_MHSA=False)
        nn = bot_stack(nn, 128 * 4, 4, strides=2, activation=activation, name="stack2_", use_MHSA=False)
        nn = bot_stack(nn, 256 * 4, 6, strides=2, activation=activation, name="stack3_", use_MHSA=False)
        nn = bot_stack(nn, 512 * 4, 3, strides=strides, activation=activation, relative_pe=relative_pe, name="stack4_", use_MHSA=True)
        return nn

    return BotNet(stack_fn, False, False, "botnet50", activation, include_top, weights, input_shape, classes, **kwargs)


def BotNet101(strides=2, activation="relu", relative_pe=True, include_top=True, weights=None, input_tensor=None, input_shape=None, classes=1000, **kwargs):
    def stack_fn(nn):
        nn = bot_stack(nn, 64 * 4, 3, strides=1, activation=activation, name="stack1_", use_MHSA=False)
        nn = bot_stack(nn, 128 * 4, 4, strides=2, activation=activation, name="stack2_", use_MHSA=False)
        nn = bot_stack(nn, 256 * 4, 23, strides=2, activation=activation, name="stack3_", use_MHSA=False)
        nn = bot_stack(nn, 512 * 4, 3, strides=strides, activation=activation, relative_pe=relative_pe, name="stack4_", use_MHSA=True)
        return nn

    return BotNet(stack_fn, False, False, "botnet101", activation, include_top, weights, input_shape, classes, **kwargs)


def BotNet152(strides=2, activation="relu", relative_pe=True, include_top=True, weights=None, input_tensor=None, input_shape=None, classes=1000, **kwargs):
    def stack_fn(nn):
        nn = bot_stack(nn, 64 * 4, 3, strides=1, activation=activation, name="stack1_", use_MHSA=False)
        nn = bot_stack(nn, 128 * 4, 8, strides=2, activation=activation, name="stack2_", use_MHSA=False)
        nn = bot_stack(nn, 256 * 4, 36, strides=2, activation=activation, name="stack3_", use_MHSA=False)
        nn = bot_stack(nn, 512 * 4, 3, strides=strides, activation=activation, relative_pe=relative_pe, name="stack4_", use_MHSA=True)
        return nn

    return BotNet(stack_fn, False, False, "botnet152", activation, include_top, weights, input_shape, classes, **kwargs)
