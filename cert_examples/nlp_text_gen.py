import tensorflow as tf

import numpy as np
import os
import time

PATH_TO_FILE: str = "datasets/nlp/shakespeare.txt"


def load_data():
    text = open(PATH_TO_FILE, "rb").read().decode(encoding="utf-8")
    vocab = sorted(set(text))

    print(f"Length of text: {len(text)} characters")
    print(text[:250])
    print(f"{len(vocab)} unique chars")
    print(vocab)


def run():
    load_data()
    print("nlp text gen")
