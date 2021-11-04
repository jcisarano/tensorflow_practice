import matplotlib.pyplot as plt
import numpy as np


def run():
    heads_probability = 0.51
    coin_tosses = (np.random.rand(10000, 10) < heads_probability).astype(np.int32)
    cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(cumulative_heads_ratio)
    plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
    plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
    plt.xlabel("Number of coin tosses")
    plt.ylabel("Heads ratio")
    plt.legend(loc="lower right")
    plt.axis([0, 10000, 0.42, 0.58])
    plt.show()