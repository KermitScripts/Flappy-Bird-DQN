import numpy as np
import matplotlib.pyplot as plt
import math
import time


def rolling_window(a, window, step_size):
    """Create a rolling window view of a numpy array."""

    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


fig = None

def episode_reward_plot(rewards, frame_idx, window_size=5, step_size=1, updating=False):
    """Plot episode rewards rolling window mean, min-max range and standard deviation.

    Parameters
    ----------
    rewards : list
        List of episode rewards.
    frame_idx : int
        Current frame index.
    window_size : int
        Rolling window size.
    step_size: int
        Step size between windows.
    updating: bool
        You can try to set updating to True, which hinders matplotlib to create a new window for every plot.
        Doesn't work with my Pycharm SciView currently.
    """

    global fig

    rewards_rolling = rolling_window(np.array(rewards), window_size, step_size)
    mean = np.mean(rewards_rolling, axis=1)
    std = np.std(rewards_rolling, axis=1)
    min_val = np.min(rewards_rolling, axis=1)
    max_val = np.max(rewards_rolling, axis=1)
    x = np.arange(math.floor(window_size / 2), len(rewards) - math.floor(window_size / 2), step_size)

    if fig is None:
        fig = plt.figure()
    
    ax = fig.add_subplot(111)

    ax.plot(x, mean, color='blue')
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, facecolor='blue')
    ax.fill_between(x, min_val, max_val, alpha=0.1, facecolor='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')

    filename  = str(frame_idx) + '.png'
    fig.savefig(filename)
