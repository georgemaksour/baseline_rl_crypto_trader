# Gym stuff
import gym
import gym_anytrading

# Stable baselines - rl stuff
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    env = gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)

    state = env.reset()
    while True:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        if done:
            print("info", info)
            break

    plt.figure(figsize=(15, 6))
    plt.cla()
    env.render_all()
    #plt.show()

    env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)
    env = DummyVecEnv([env_maker])

    model = A2C('MlpLstmPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)


    env = gym.make('stocks-v0', df=df, frame_bound=(90, 110), window_size=5)
    obs = env.reset()
    while True:
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("info", info)
            break

    plt.figure(figsize=(15, 6))
    plt.cla()
    env.render_all()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
