# Transfer learning with TensorFlow part 1: Feature Extraction
# Transfer learning leverages a working model's existing architecture and learned patterns for a similar problem

# Two main benefits:
#   1. Uses existing NN architecture proven to work on a problem similar to our own
#   2. Improves performance by starting with working NN

import data_utils as du
import tf_hub_example as tfhe
import tf_hub_mobilenet as mobilenet


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mobilenet.run()
    # tfhe.run()
    # du.list_filecount_in_dir(dir=du.LOCAL_DATA_PATH)
    # du.load_and_prep_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
