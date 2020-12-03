import tensorflow as tf
import numpy as np
import os
import tensorflow.keras.backend as K


def scale_softmax(y_true, y_pred, scale=64.0, from_logits=False, label_smoothing=0):
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred * scale, from_logits=from_logits, label_smoothing=label_smoothing
    )


def margin_softmax(y_true, y_pred, power=2, scale=0.4, from_logits=False, label_smoothing=0):
    margin_soft = tf.where(tf.cast(y_true, dtype=tf.bool), (y_pred ** power + y_pred * scale) / 2, y_pred)
    return tf.keras.losses.categorical_crossentropy(
        y_true, margin_soft, from_logits=from_logits, label_smoothing=label_smoothing
    )


# margin_softmax class wrapper
class MarginSoftmax(tf.keras.losses.Loss):
    def __init__(self, power=2, scale=0.4, from_logits=False, label_smoothing=0, **kwargs):
        super(MarginSoftmax, self).__init__(**kwargs)
        self.power, self.scale, self.from_logits, self.label_smoothing = power, scale, from_logits, label_smoothing

    def call(self, y_true, y_pred):
        margin_soft = tf.where(tf.cast(y_true, dtype=tf.bool), (y_pred ** self.power + y_pred * self.scale) / 2, y_pred)
        return tf.keras.losses.categorical_crossentropy(
            y_true, margin_soft, from_logits=self.from_logits, label_smoothing=self.label_smoothing
        )

    def get_config(self):
        config = super(MarginSoftmax, self).get_config()
        config.update(
            {
                "power": self.power,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Single function one
def arcface_loss(y_true, y_pred, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, from_logits=True, label_smoothing=0):
    # norm_logits = y_pred[:, 512:]
    threshold = np.cos((np.pi - margin2) / margin1)
    theta_min = (-1 - margin3) * 2
    norm_logits = y_pred
    y_pred_vals = norm_logits[tf.cast(y_true, dtype=tf.bool)]
    # y_pred_vals = tf.clip_by_value(y_pred_vals, clip_value_min=-1.0, clip_value_max=1.0)
    if margin1 == 1.0 and margin3 == 0.0:
        theta = tf.cos(tf.acos(y_pred_vals) + margin2)
    else:
        theta = tf.cos(tf.acos(y_pred_vals) * margin1 + margin2) - margin3
        # Grad(theta) == 0
        #   ==> np.sin(np.math.acos(xx) * margin1 + margin2) == 0
        #   ==> np.math.acos(xx) * margin1 + margin2 == np.pi
        #   ==> xx == np.cos((np.pi - margin2) / margin1)
        #   ==> theta_min == -1 - margin3
    # theta_valid = tf.where(theta < y_pred_vals, theta, y_pred_vals)
    theta_valid = tf.where(y_pred_vals > threshold, theta, theta_min - theta)
    theta_one_hot = tf.expand_dims(theta_valid - y_pred_vals, 1) * tf.cast(y_true, dtype=tf.float32)
    arcface_logits = (theta_one_hot + norm_logits) * scale
    # tf.assert_equal(tf.math.is_nan(tf.reduce_mean(arcface_logits)), False)
    return tf.keras.losses.categorical_crossentropy(
        y_true, arcface_logits, from_logits=from_logits, label_smoothing=label_smoothing
    )


# ArcfaceLoss class
class ArcfaceLoss(tf.keras.losses.Loss):
    def __init__(self, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        # reduction = tf.keras.losses.Reduction.NONE if tf.distribute.has_strategy() else tf.keras.losses.Reduction.AUTO
        # super(ArcfaceLoss, self).__init__(**kwargs, reduction=reduction)
        super(ArcfaceLoss, self).__init__(**kwargs)
        self.margin1, self.margin2, self.margin3, self.scale = margin1, margin2, margin3, scale
        self.from_logits, self.label_smoothing = from_logits, label_smoothing
        self.threshold = np.cos((np.pi - margin2) / margin1)  # grad(theta) == 0
        self.theta_min = (-1 - margin3) * 2
        # self.reduction_func = tf.keras.losses.CategoricalCrossentropy(
        #     from_logits=from_logits, label_smoothing=label_smoothing, reduction=reduction
        # )

    def call(self, y_true, norm_logits):
        # norm_logits = y_pred
        pick_cond = tf.cast(y_true, dtype=tf.bool)
        y_pred_vals = norm_logits[pick_cond]
        if self.margin1 == 1.0 and self.margin3 == 0.0:
            theta = tf.cos(tf.acos(y_pred_vals) + self.margin2)
        else:
            theta = tf.cos(tf.acos(y_pred_vals) * self.margin1 + self.margin2) - self.margin3
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        theta_one_hot = tf.expand_dims(theta_valid - y_pred_vals, 1) * tf.cast(y_true, dtype=tf.float32)
        arcface_logits = (theta_one_hot + norm_logits) * self.scale
        # theta_one_hot = tf.expand_dims(theta_valid, 1) * tf.cast(y_true, dtype=tf.float32)
        # arcface_logits = tf.where(pick_cond, theta_one_hot, norm_logits) * self.scale
        return tf.keras.losses.categorical_crossentropy(
            y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing
        )
        # return self.reduction_func(y_true, arcface_logits)

    def get_config(self):
        config = super(ArcfaceLoss, self).get_config()
        config.update(
            {
                "margin1": self.margin1,
                "margin2": self.margin2,
                "margin3": self.margin3,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Mxnet insightface loss_type == 4, just same effect as `arcface_loss` [margin1=1.0, margin2=0.5, margin3=0.0]
class ArcfaceLossT4(tf.keras.losses.Loss):
    def __init__(self, margin=0.5, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(ArcfaceLossT4, self).__init__(**kwargs)
        self.margin, self.scale, self.from_logits, self.label_smoothing = margin, scale, from_logits, label_smoothing
        self.margin_cos, self.margin_sin = tf.cos(margin), tf.sin(margin)
        self.threshold = tf.cos(np.pi - margin)
        self.low_pred_punish = tf.sin(np.pi - margin) * margin

    def call(self, y_true, norm_logits):
        pick_cond = tf.cast(y_true, dtype=tf.bool)
        y_pred_vals = norm_logits[pick_cond]
        theta = y_pred_vals * self.margin_cos - tf.sqrt(1 - tf.pow(y_pred_vals, 2)) * self.margin_sin
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, y_pred_vals - self.low_pred_punish)
        # theta_one_hot = tf.expand_dims(theta_valid, 1) * tf.cast(y_true, dtype=tf.float32)
        # arcface_logits = tf.where(pick_cond, theta_one_hot, norm_logits) * self.scale
        theta_one_hot = tf.expand_dims(theta_valid - y_pred_vals, 1) * tf.cast(y_true, dtype=tf.float32)
        arcface_logits = (theta_one_hot + norm_logits) * self.scale
        return tf.keras.losses.categorical_crossentropy(
            y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing
        )

    def get_config(self):
        config = super(ArcfaceLossT4, self).get_config()
        config.update(
            {
                "margin": self.margin,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ArcfaceLoss class
# [AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations](https://arxiv.org/pdf/1905.00292.pdf)
class AdaCosLossSimple(tf.keras.losses.Loss):
    def __init__(self, num_classes, scale=0, max_median=np.pi / 4, from_logits=True, label_smoothing=0, **kwargs):
        super(AdaCosLossSimple, self).__init__(**kwargs)
        self.max_median, self.from_logits, self.label_smoothing = max_median, from_logits, label_smoothing
        self.num_classes = num_classes
        self.theta_med_max = tf.cast(max_median, 'float32')
        if scale == 0:
            self.scale = tf.sqrt(2.) * tf.math.log(float(num_classes) - 1)
        else:
            # In reload condition
            self.scale = tf.cast(scale, 'float32')

    @tf.function
    def call(self, y_true, norm_logits):
        pick_cond = tf.cast(y_true, dtype=tf.bool)
        y_pred_vals = norm_logits[pick_cond]
        theta = tf.acos(y_pred_vals)
        med_pos = tf.shape(norm_logits)[0] // 2 - 1
        theta_med = tf.sort(theta)[med_pos]

        B_avg = tf.where(pick_cond, tf.zeros_like(norm_logits), tf.exp(self.scale * norm_logits))
        B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1))
        self.scale = tf.math.log(B_avg) / tf.cos(tf.minimum(self.theta_med_max, theta_med))
        tf.print(", scale =", self.scale, ", theta_med =", theta_med, end='')

        arcface_logits = norm_logits * self.scale
        return tf.keras.losses.categorical_crossentropy(
            y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing
        )
        # return self.reduction_func(y_true, arcface_logits)

    def get_config(self):
        config = super(AdaCosLossSimple, self).get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                # "scale": self.scale.numpy(),
                "max_median": self.max_median,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AdaCosLossT4(tf.keras.losses.Loss):
    def __init__(self, batch_size, num_classes, margin=0.5, scale=0, max_median=np.cos(np.pi / 4), from_logits=True, label_smoothing=0, **kwargs):
        super(AdaCosLossT4, self).__init__(**kwargs)
        self.margin, self.max_median, self.from_logits, self.label_smoothing = margin, max_median, from_logits, label_smoothing
        self.batch_size, self.num_classes = batch_size, num_classes
        self.threshold = tf.cos(np.pi - margin)
        self.low_pred_punish = tf.sin(np.pi - margin) * margin

        margin_cos, margin_sin = tf.cos(margin), tf.sin(margin)
        self.theta_func = lambda xx: xx * margin_cos - tf.sqrt(1 - tf.pow(xx, 2)) * margin_sin

        # self.theta_med_max = self.theta_func(tf.cast(max_median, 'float32'))
        self.theta_med_max = tf.cast(max_median, 'float32')
        self.med_pos = batch_size // 2
        if scale == 0:
            self.scale = tf.sqrt(2.) * tf.math.log(float(num_classes) - 1)
        else:
            # In reload condition
            self.scale = tf.cast(scale, 'float32')
        tf.print("Init med_pos:", self.med_pos, "scale:", self.scale, "theta_med_max:", self.theta_med_max)

    @tf.function
    def call(self, y_true, norm_logits):
        pick_cond = tf.cast(y_true, dtype=tf.bool)
        y_pred_vals = norm_logits[pick_cond]
        theta_med = tf.sort(y_pred_vals)[self.med_pos]

        B_avg = tf.where(pick_cond, tf.zeros_like(norm_logits), tf.exp(self.scale * norm_logits))
        B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1))
        self.scale = tf.math.log(B_avg) / tf.maximum(self.theta_med_max, theta_med)
        tf.print(", scale =", self.scale, ", theta_med =", theta_med, end='')

        theta = self.theta_func(y_pred_vals)
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, y_pred_vals - self.low_pred_punish)

        # theta_one_hot = tf.expand_dims(theta_valid, 1) * tf.cast(y_true, dtype=tf.float32)
        # arcface_logits = tf.where(pick_cond, theta_one_hot, norm_logits) * self.scale
        theta_one_hot = tf.expand_dims(theta_valid - y_pred_vals, 1) * tf.cast(y_true, dtype=tf.float32)
        arcface_logits = (theta_one_hot + norm_logits) * self.scale
        return tf.keras.losses.categorical_crossentropy(
            y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing
        )

    def get_config(self):
        config = super(AdaCosLossT4, self).get_config()
        config.update(
            {
                "batch_size": self.batch_size,
                "num_classes": self.num_classes,
                "margin": self.margin,
                # "scale": self.scale.numpy(),
                "max_median": self.max_median,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Callback to save center values on each epoch end
class Save_Numpy_Callback(tf.keras.callbacks.Callback):
    def __init__(self, save_file, save_tensor):
        super(Save_Numpy_Callback, self).__init__()
        self.save_file = os.path.splitext(save_file)[0]
        self.save_tensor = save_tensor

    def on_epoch_end(self, epoch=0, logs=None):
        np.save(self.save_file, self.save_tensor.numpy())


class CenterLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, emb_shape=512, alpha=0.5, initial_file=None, **kwargs):
        super(CenterLoss, self).__init__(**kwargs)
        self.num_classes, self.emb_shape, self.alpha = num_classes, emb_shape, alpha
        self.initial_file = initial_file
        # centers = tf.Variable(tf.zeros([num_classes, emb_shape]), trainable=False)
        centers = tf.Variable(tf.zeros([num_classes, emb_shape]), trainable=False, aggregation=tf.VariableAggregation.MEAN)
        if initial_file:
            if os.path.exists(initial_file):
                print(">>>> Reload from center backup:", initial_file)
                aa = np.load(initial_file)
                centers.assign(aa)
            self.save_centers_callback = Save_Numpy_Callback(initial_file, centers)
        self.centers = centers
        if tf.distribute.has_strategy():
            self.num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        else:
            self.num_replicas = 1

    def call(self, y_true, embedding):
        # embedding = y_pred[:, : self.emb_shape]
        labels = tf.argmax(y_true, axis=1)
        centers_batch = tf.gather(self.centers, labels)
        # loss = tf.reduce_mean(tf.square(embedding - centers_batch))
        loss = tf.reduce_sum(tf.square(embedding - centers_batch), axis=-1)

        # Update centers
        # diff = (1 - self.alpha) * (centers_batch - embedding)
        diff = centers_batch - embedding
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = self.num_replicas * self.alpha * diff
        # print(centers_batch.shape, self.centers.shape, labels.shape, diff.shape)
        self.centers.assign(tf.tensor_scatter_nd_sub(self.centers, tf.expand_dims(labels, 1), diff))
        # centers_batch = tf.gather(self.centers, labels)
        return loss

    def get_config(self):
        config = super(CenterLoss, self).get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "emb_shape": self.emb_shape,
                "alpha": self.alpha,
                "initial_file": self.initial_file,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "feature_dim" in config:
            config["emb_shape"] = config.pop("feature_dim")
        if "factor" in config:
            config.pop("factor")
        if "logits_loss" in config:
            config.pop("logits_loss")
        return cls(**config)


def batch_hard_triplet_loss(labels, embeddings, alpha=0.35):
    labels = tf.argmax(labels, axis=1)
    # labels = tf.squeeze(labels)
    # labels.set_shape([None])
    pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    norm_emb = tf.nn.l2_normalize(embeddings, 1)
    dists = tf.matmul(norm_emb, tf.transpose(norm_emb))
    # pos_dists = tf.ragged.boolean_mask(dists, pos_mask)
    pos_dists = tf.where(pos_mask, dists, tf.ones_like(dists))
    hardest_pos_dist = tf.reduce_min(pos_dists, -1)
    # neg_dists = tf.ragged.boolean_mask(dists, tf.logical_not(pos_mask))
    neg_dists = tf.where(pos_mask, tf.ones_like(dists) * -1, dists)
    hardest_neg_dist = tf.reduce_max(neg_dists, -1)
    basic_loss = hardest_neg_dist - hardest_pos_dist + alpha
    # return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return tf.maximum(basic_loss, 0.0)


def batch_all_triplet_loss(labels, embeddings, alpha=0.35):
    labels = tf.argmax(labels, axis=1)
    # labels = tf.squeeze(labels)
    # labels.set_shape([None])
    pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    norm_emb = tf.nn.l2_normalize(embeddings, 1)
    dists = tf.matmul(norm_emb, tf.transpose(norm_emb))

    pos_dists = tf.where(pos_mask, dists, tf.ones_like(dists))
    pos_dists_loss = tf.reduce_sum(1.0 - pos_dists, -1) / tf.reduce_sum(tf.cast(pos_mask, dtype=tf.float32), -1)
    hardest_pos_dist = tf.expand_dims(tf.reduce_min(pos_dists, -1), 1)

    neg_valid_mask = tf.logical_and(tf.logical_not(pos_mask), (hardest_pos_dist - dists) < alpha)
    neg_dists_valid = tf.where(neg_valid_mask, dists, tf.zeros_like(dists))
    neg_dists_loss = tf.reduce_sum(neg_dists_valid, -1) / (tf.reduce_sum(tf.cast(neg_valid_mask, dtype=tf.float32), -1) + 1)
    return pos_dists_loss + neg_dists_loss


# TripletLoss helper class definitions
class TripletLossWapper(tf.keras.losses.Loss):
    def __init__(self, triplet_loss_func, alpha, **kwargs):
        # reduction = tf.keras.losses.Reduction.NONE if tf.distribute.has_strategy() else tf.keras.losses.Reduction.AUTO
        # super(TripletLossWapper, self).__init__(**kwargs, reduction=reduction)
        super(TripletLossWapper, self).__init__(**kwargs)
        self.alpha, self.triplet_loss_func = alpha, triplet_loss_func

    def call(self, y_true, y_pred):
        return self.triplet_loss_func(y_true, y_pred, self.alpha)

    def get_config(self):
        config = super(TripletLossWapper, self).get_config()
        config.update({"alpha": self.alpha})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BatchHardTripletLoss(TripletLossWapper):
    def __init__(self, alpha, **kwargs):
        super(BatchHardTripletLoss, self).__init__(batch_hard_triplet_loss, alpha, **kwargs)


class BatchAllTripletLoss(TripletLossWapper):
    def __init__(self, alpha, **kwargs):
        super(BatchAllTripletLoss, self).__init__(batch_hard_triplet_loss, alpha, **kwargs)


def distiller_loss(true_emb_normed, pred_emb):
    pred_emb_normed = tf.nn.l2_normalize(pred_emb, axis=-1)
    loss = tf.reduce_sum(tf.square(true_emb_normed - pred_emb_normed), axis=-1)
    return loss
