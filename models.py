import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

tf.compat.v1.disable_eager_execution()  # usually using this for fastest performance


class Actor_Model:
    def __init__(self, input_shape, action_shape) -> None:
        self.input_shape = input_shape
        self.action_space = action_shape
        self.model = self._make_model(
            input_shape=input_shape, action_shape=action_shape
        )

    def _make_model(self, input_shape, action_shape):
        x_input = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.Flatten(input_shape=input_shape)(x_input)
        x = tf.keras.layers.Dense(units=512, activation="relu")(x)
        x = tf.keras.layers.Dense(units=256, activation="relu")(x)
        x = tf.keras.layers.Dense(units=64, activation="relu")(x)
        out = tf.keras.layers.Dense(units=action_shape, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=x_input, outputs=out)
        model.compile(
            loss=self._ppo_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        )
        return model

    def _ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = (
            y_true[:, :1],
            y_true[:, 1 : 1 + self.action_space],
            y_true[:, 1 + self.action_space :],
        )
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = (
            K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING)
            * advantages
        )

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        return self.model.predict(state)


class Critic_Model:
    def __init__(self, input_shape) -> None:
        self.model = self._make_model(input_shape=input_shape)

    def _make_model(self, input_shape):
        x_input = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.Flatten(input_shape=input_shape)(x_input)
        x = tf.keras.layers.Dense(units=512, activation="relu")(x)
        x = tf.keras.layers.Dense(units=256, activation="relu")(x)
        x = tf.keras.layers.Dense(units=64, activation="relu")(x)
        out = tf.keras.layers.Dense(units=1, activation=None)(x)

        model = tf.keras.models.Model(inputs=x_input, outputs=out)
        model.compile(
            loss=self._ppo_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        )
        return model

    def _ppo_loss(self, y_true, y_pred):
        return K.mean((y_true - y_pred) ** 2)

    def predict(self, state):
        return self.model.predict([state, np.zeros((state.shape[0], 1))])
