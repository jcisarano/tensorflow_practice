"""Transfer Learning with TensorFlow 3 - Scaling Up

    Using transfer learning and feature extraction, we will now train on all 101 classes of the Food101 dataset.
    The goal is to beat the performance of the original Food101 paper, but using only 10% of the training set
    by leveraging the power of transfer learning.
"""

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_histories, walk_through_dir

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Transfer Learning')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
