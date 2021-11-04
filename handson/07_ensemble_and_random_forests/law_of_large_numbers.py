import matplotlib.pyplot as plt
import numpy as np


def run():
    heads_probability = 0.51
    # produce 10x10000 matrix to represent 10 coins tossed 10000 times
    # coin_tosses.shape = (10000, 10)
    coin_tosses = (np.random.rand(10000, 10) < heads_probability).astype(np.int32)
    # np.cumsum along axis 0 adds up cols of 0/1 vals
    # np.arange returns evenly spaced values over range 1, 10001
    # np.reshape makes array shape (10000, 1)
    # so each row is divided by its index+1: row 0 divided by 1, row 1 divided by 2, etc
    cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)

    plt.figure(figsize=(8, 4.5))
    # plots all the coin toss values
    plt.plot(cumulative_heads_ratio)

    # draw the horizontal lines
    plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
    plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")

    plt.xlabel("Number of coin tosses")
    plt.ylabel("Heads ratio")
    plt.legend(loc="lower right")

    # make the axes fit the data
    plt.axis([0, 10000, 0.42, 0.58])
    plt.show()
