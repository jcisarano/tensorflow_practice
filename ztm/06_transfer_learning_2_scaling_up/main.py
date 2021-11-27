"""Transfer Learning with TensorFlow 3 - Scaling Up

    Using transfer learning and feature extraction, we will now train on all 101 classes of the Food101 dataset.
    The goal is to beat the performance of the original Food101 paper, but using only 10% of the training set
    by leveraging the power of transfer learning.
"""

import food_classifier as fc
import feature_extraction_exercise as fee

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fee.run()
    # fc.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
