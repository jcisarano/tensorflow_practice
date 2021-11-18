
import data_utils as du
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

def run():
    walk_through_dir(du.LOCAL_DATA_PATH_1_PERCENT)