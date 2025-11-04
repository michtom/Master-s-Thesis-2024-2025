import tensorflow as tf
from typing import Callable


def huber_with_varmatch(delta=1.0, lam=0.1) -> Callable:
    huber = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def loss(y_true, y_pred):
        l = huber(y_true, y_pred)
        std_t = tf.math.reduce_std(y_true)
        std_p = tf.math.reduce_std(y_pred)
        vpen = tf.square(std_p - std_t)
        return tf.reduce_mean(l) + lam * vpen

    return loss


def expectile_loss(y_true, y_pred, tau=0.62) -> Callable:
    err = y_true - y_pred
    w = tf.where(err >= 0.0, tau, 1.0 - tau)
    return tf.reduce_mean(w * tf.square(err))


def loss_expectile_var(tau=0.62, lam=0.15) -> Callable:
    def loss(y_true, y_pred):
        l = expectile_loss(y_true, y_pred, tau)
        std_t = tf.math.reduce_std(y_true) + 1e-8
        std_p = tf.math.reduce_std(y_pred) + 1e-8
        return l + lam * tf.square(std_p - std_t)
    return loss
