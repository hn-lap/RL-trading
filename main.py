from collections import deque

import numpy as np
import pandas as pd

from envs.stock_envs import Envs


def random_games(env, visualize, train_episodes=50):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action = np.random.randint(3, size=1)[0]
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", episode, env.net_worth)
                break

    print("average {} episodes random net_worth: {}".format(train_episodes, average_net_worth / train_episodes))


def train_agent(env, visualize=False, train_episodes=50, training_batch_size=500):
    env.create_tracking_log()  # create TensorBoard writer
    total_average = deque(maxlen=100)  # save recent 100 episodes net worth
    best_average = 0  # used to track best average net worth
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        states, actions, rewards, predictions, dones, next_states = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for t in range(training_batch_size):
            action, prediction = env.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state

        env.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.net_worth)
        average = np.average(total_average)

        env.writer.add_scalar("Data/average net_worth", average, episode)
        env.writer.add_scalar("Data/episode_orders", env.episode_orders, episode)
        print(
            "Steps: {}/{}--Net-worth:{:.2f}--Average:{:.2f}--episode_orders:{}".format(
                episode, train_episodes, env.net_worth, average, env.episode_orders
            )
        )
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                env.save()


def test_agent(env, visualize=True, test_episodes=10):
    env.load()  # load the model
    average_net_worth = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action, prediction = env.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", episode, env.net_worth, env.episode_orders)
                break

    print("average {} episodes agent net_worth: {}".format(test_episodes, average_net_worth / test_episodes))


if __name__ == "__main__":
    df = pd.read_csv("/home/eco0936_namnh/CODE/tradebot/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_2/pricedata.csv")
    df = df.sort_values("Date")

    lookback_window_size = 50
    train_df = df[: -720 - lookback_window_size]
    test_df = df[-720 - lookback_window_size :]  # 30 days

    train_env = Envs(
        train_df,
        initial_balance=1000,
        lookback_windowsize=lookback_window_size,
        render_range=100,
    )
    test_env = Envs(
        test_df,
        initial_balance=1000,
        lookback_windowsize=lookback_window_size,
        render_range=100,
    )

    # train_agent(train_env, visualize=True, train_episodes=1000, training_batch_size=500)
    test_agent(test_env, visualize=True, test_episodes=500)
    # random_games(test_env, visualize=True, train_episodes=500)
