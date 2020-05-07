import tensorflow as tf
import numpy as np
import os


def scale_softmax(y_true, y_pred, scale=64.0, from_logits=True, label_smoothing=0):
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred * scale, from_logits=from_logits, label_smoothing=label_smoothing
    )


def margin_softmax(y_true, y_pred, power=2, scale=0.4, from_logits=False, label_smoothing=0):
    margin_soft = tf.where(tf.cast(y_true, dtype=tf.bool), (y_pred ** power + y_pred * scale) / 2, y_pred)
    return tf.keras.losses.categorical_crossentropy(
        y_true, margin_soft, from_logits=from_logits, label_smoothing=label_smoothing
    )


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
        super(ArcfaceLoss, self).__init__(**kwargs)
        self.margin1, self.margin2, self.margin3, self.scale = margin1, margin2, margin3, scale
        self.from_logits, self.label_smoothing = from_logits, label_smoothing
        self.threshold = np.cos((np.pi - margin2) / margin1)  # grad(theta) == 0
        self.theta_min = (-1 - margin3) * 2

    def call(self, y_true, y_pred):
        norm_logits = y_pred
        y_pred_vals = norm_logits[tf.cast(y_true, dtype=tf.bool)]
        if self.margin1 == 1.0 and self.margin3 == 0.0:
            theta = tf.cos(tf.acos(y_pred_vals) + self.margin2)
        else:
            theta = tf.cos(tf.acos(y_pred_vals) * self.margin1 + self.margin2) - self.margin3
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        theta_one_hot = tf.expand_dims(theta_valid - y_pred_vals, 1) * tf.cast(y_true, dtype=tf.float32)
        arcface_logits = (theta_one_hot + norm_logits) * self.scale
        return tf.keras.losses.categorical_crossentropy(
            y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing
        )

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


# Simplified one
def arcface_loss_2(y_true, y_pred, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, from_logits=True):
    theta = tf.cos(tf.acos(y_pred) * margin1 + margin2) - margin3
    cond = tf.logical_and(tf.cast(y_true, dtype=tf.bool), theta < y_pred)
    arcface_logits = tf.where(cond, theta, y_pred) * scale
    return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=from_logits)


# Callback to save center values on each epoch end
class Save_Numpy_Callback(tf.keras.callbacks.Callback):
    def __init__(self, save_file, save_tensor):
        super(Save_Numpy_Callback, self).__init__()
        self.save_file = os.path.splitext(save_file)[0]
        self.save_tensor = save_tensor

    def on_epoch_end(self, epoch=0, logs=None):
        np.save(self.save_file, self.save_tensor.numpy())


class CenterLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, feature_dim=512, alpha=0.5, factor=1.0, initial_file=None, logits_loss=None, **kwargs):
        super(CenterLoss, self).__init__(**kwargs)
        self.num_classes, self.feature_dim, self.alpha, self.factor = num_classes, feature_dim, alpha, factor
        self.initial_file = initial_file
        centers = tf.Variable(tf.zeros([num_classes, feature_dim]), trainable=False)
        if initial_file:
            if os.path.exists(initial_file):
                aa = np.load(initial_file)
                centers.assign(aa)
            self.save_centers_callback = Save_Numpy_Callback(initial_file, centers)
        self.centers = centers
        self.logits_loss = logits_loss

    def call(self, y_true, y_pred):
        embedding = y_pred[:, : self.feature_dim]
        labels = tf.argmax(y_true, axis=1)
        centers_batch = tf.gather(self.centers, labels)
        # loss = tf.reduce_mean(tf.square(embedding - centers_batch))
        loss = tf.reduce_mean(tf.square(embedding - centers_batch), axis=-1)

        # Update centers
        # diff = (1 - self.alpha) * (centers_batch - embedding)
        diff = centers_batch - embedding
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = self.alpha * diff
        # print(centers_batch.shape, self.centers.shape, labels.shape, diff.shape)
        self.centers.assign(tf.tensor_scatter_nd_sub(self.centers, tf.expand_dims(labels, 1), diff))
        # centers_batch = tf.gather(self.centers, labels)
        if self.logits_loss:
            return self.logits_loss(y_true, y_pred[:, self.feature_dim :]) + loss * self.factor
        else:
            return loss * self.factor

    def accuracy(self, y_true, y_pred):
        """ Accuracy function for logits only """
        logits = y_pred[:, self.feature_dim :]
        return tf.keras.metrics.categorical_accuracy(y_true, logits)

    def get_config(self):
        config = super(CenterLoss, self).get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "feature_dim": self.feature_dim,
                "alpha": self.alpha,
                "factor": self.factor,
                "initial_file": self.initial_file,
                "logits_loss": self.logits_loss,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def batch_hard_triplet_loss(labels, embeddings, alpha=0.35):
    labels = tf.squeeze(labels)
    labels.set_shape([None])
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
    return tf.reduce_mean(tf.maximum(basic_loss, 0.0))


def batch_all_triplet_loss(labels, embeddings, alpha=0.35):
    labels = tf.squeeze(labels)
    labels.set_shape([None])
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


# Two helper class definitions
class BatchHardTripletLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, **kwargs):
        super(BatchHardTripletLoss, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return batch_hard_triplet_loss(y_true, y_pred, self.alpha)

    def get_config(self):
        config = super(BatchHardTripletLoss, self).get_config()
        config.update({"alpha": self.alpha})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BatchAllTripletLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, **kwargs):
        super(BatchAllTripletLoss, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return batch_all_triplet_loss(y_true, y_pred, self.alpha)

    def get_config(self):
        config = super(BatchHardTripletLoss, self).get_config()
        config.update({"alpha": self.alpha})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
