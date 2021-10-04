import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import simple_moving_average


def plot_history(history, smoothing=1):
    _, ax = plt.subplots(1, 3, figsize=(20, 4))
    ax[0].plot(simple_moving_average(history['regrets'], smoothing))
    ax[0].set_title("Smoothed regret vs time_step")

    cumulative_regret = np.cumsum(history['regrets'])
    x = np.arange(1, len(cumulative_regret) + 1)
    log_reference = np.log(x)
    log_reference *= cumulative_regret[-1] / log_reference[-1]
    ax[1].plot(x, cumulative_regret, label='regret')
    ax[1].plot(x, log_reference, label='log_reference')
    ax[1].set_title("Cumulative regret")
    ax[1].legend()
    sns.heatmap(history['action_values'].T, ax=ax[2])
    ax[2].set_xticks([])
    ax[2].set_title("Estimated action values over time")
