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


def parse_file(filepath):
    lines = get_lines(filepath)
    parsed_lines = []
    for line in lines:
        line = line.replace("\n", "")
        if line.startswith("###"):
            line = line.replace("###", "")
            abstract = {"line_number": line, "total_lines": 0}
        elif len(line) == 0:
            parsed_lines.append(abstract)
            continue
        else:
            split = line.split("\t")
            abstract["target"] = split[0]
            abstract["text"] = split[1]
            abstract["total_lines"] = abstract["total_lines"] + 1

    print(len(parsed_lines))
    print(parsed_lines[:5])


def run():
    parse_file(DATA_DIR_20K_NUM_REPL + "train.txt")
    # filenames = [DATA_DIR_20K_NUM_REPL + filename for filename in os.listdir(DATA_DIR_20K_NUM_REPL)]
    # print(filenames)

    # train_lines = get_lines(DATA_DIR_20K_NUM_REPL + "train.txt")
    # print(train_lines[:30])
    # print(len(train_lines))
    # start the experiments using 20k dataset with numbers replaced by @ sign
    print("skim lit")