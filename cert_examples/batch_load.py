from transfer_learning import load_and_prep_data
# import tensorflow_datasets as tfds


# def tfds_load_example():
#     datasets_list = tfds.list_builders()



def run():
    # loads in batches
    train_data, test_data = load_and_prep_data()


    print("batch load")
