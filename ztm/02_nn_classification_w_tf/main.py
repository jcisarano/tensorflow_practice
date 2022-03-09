# Intro to NN classification with TensorFlow
# How to write NN for classification problems, where you need to classify something as one thing or another
# Types of classification:
# Binary classification
# Multiclass classification
# Multilabel classification

from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import examine_circles_data

import simple_model as sm

import evaluation as eval

import multiclass_classification as mc

import playground_copy as pc

import moon_data as md

import fashion_prediction_exercise as fpe

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # make 1000 examples
    X, y = examine_circles_data.generate_circles()
    # examine_circles_data.examine_data(X, y)

    # Steps to build NN model
    # 1. Prepare the data
    # 2. Build the model: inputs, outputs, layers
    # 3. Compile the model: loss function, optimizer, metrics
    # 4. Fit the model to the training data
    # 5. Evaluate and improve through experimentation
    # sm.run(X, y)

    # Remember the three datasets:
    # training
    # validation
    # eval.run(X, y)

    # mc.run()
    # pc.run(X, y)
    md.run()

    # fpe.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
