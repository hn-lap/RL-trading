import copy

import numpy as np
import tensorflow as tf

from custom_agent.models import Actor_Model, Critic_Model


class Agent:
    def __init__(self, state_dim) -> None:
        self.state_size = (state_dim, 4)
        self.action_space = np.array([0, 1])
        self.actor_model, self.critic_model = self._load_model()

    def _load_model(self):
        actor_model = Actor_Model(
            input_shape=self.state_size, action_shape=self.action_space.shape[0]
        )
        critic_model = Critic_Model(input_shape=self.state_size)
        return actor_model, critic_model

    def act(self, state):
        prediction = self.actor_model.predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def get_gaes(
        self,
        rewards,
        dones,
        values,
        next_values,
        gamma=0.99,
        lamda=0.95,
        normalize=True,
    ):
        deltas = [
            r + gamma * (1 - d) * nv - v
            for r, d, nv, v in zip(rewards, dones, next_values, values)
        ]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)
        next_states = np.vstack(next_states)

        values = self.critic_model.predict(states)
        next_values = self.critic_model.predict(next_states)

        advantages, target = self.get_gaes(
            rewards, dones, np.squeeze(values), np.squeeze(next_values)
        )
        y_true = np.hstack([advantages, predictions, actions])

        a_loss = self.actor_model.fit(states, y_true, epochs=1, verbose=0, shuffle=True)
        c_loss = self.critic_model.fit(
            states, target, epochs=1, verbose=0, shuffle=True
        )

        return a_loss, c_loss

    def save(self, name="trader"):
        # save keras model weights
        self.actor_model.saved_weights(f"{name}_Actor.h5")
        self.critic_model.saved_weights(f"{name}_Critic.h5")

    def load(self, name="trader"):
        # load keras model weights
        self.actor_model.loading_weights(f"{name}_Actor.h5")
        self.critic_model.loading_weights(f"{name}_Critic.h5")
