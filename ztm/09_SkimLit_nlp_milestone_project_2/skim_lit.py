import os

DATA_DIR_20K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"

"""
    Desired structure for data is list of dictionaries:
    [{
        'line_number' : 0,
        'target': 'BACKGROUND',
        'text': 'Emotional eating is associated with overeating and the development of obesity .',
        'total_lines': 11
    },
    ...
    ]
"""


def get_lines(filepath):
    """
    Load text file and returns lines in that file as list

    :return:
        List of strings with one string per line from target file
    """
    with open(filepath, "r") as f:
        return f.readlines()


def run():
    filenames = [DATA_DIR_20K_NUM_REPL + filename for filename in os.listdir(DATA_DIR_20K_NUM_REPL)]
    print(filenames)

    train_lines = get_lines(DATA_DIR_20K_NUM_REPL + "train.txt")
    print(train_lines[:30])
    print(len(train_lines))
    # start the experiments using 20k dataset with numbers replaced by @ sign
    print("skim lit")