import os

DATA_DIR_20K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
DATA_DIR_200K_NUM_REPL: str = "dataset/pubmed-rct-master/PubMed_200k_RCT_numbers_replaced_with_at_sign/"


def get_lines(filepath):
    """
    Load text file and returns lines in that file as list

    :return:
        List of strings with one string per line from target file
    """
    with open(filepath, "r") as f:
        return f.readlines()


def preprocess_text_with_line_numbers(filepath):
    """
    Returns list of dictionaries of abstract line data. Sample data format:
        [{
            'line_number' : 0,
            'target': 'BACKGROUND',
            'text': 'Emotional eating is associated with overeating and the development of obesity .',
            'total_lines': 11
        },
        ...
        ]
    :param filepath:
    :return:
    """
    input_lines = get_lines(filepath)
    abstract_lines = ""
    abstract_samples = []

    for line in input_lines:
        if line.startswith("###"):  # this is an id line, the start of an abstract
            abstract_id = line
            abstract_lines = ""
        elif line.isspace():  # this is an empty line, the end of an abstract
            abstract_line_split = abstract_lines.splitlines()

            # now iterate through line in a single abstract to handle and count them:
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t")  # splits id from text
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number  # what number is this line in the abstract?
                line_data["total_lines"] = len(abstract_line_split)-1
                abstract_samples.append(line_data)
        else:
            abstract_lines += line

    return abstract_samples


def parse_file(filepath):
    """
    My version of the preprocess_text_with_line_numbers() task.

    :param filepath:
    :return:
    """
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
            abstract["total_lines"] += 1

    print(len(parsed_lines))
    print(parsed_lines[:5])

    return parsed_lines


def run():
    train_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "train.txt")
    val_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "dev.txt")
    test_samples = preprocess_text_with_line_numbers(filepath=DATA_DIR_20K_NUM_REPL + "test.txt")
    print(train_samples[:10])
    print(len(train_samples))
    print(len(val_samples))
    print(len(test_samples))
    train_samples = parse_file(DATA_DIR_20K_NUM_REPL + "train.txt")

    # train_lines = get_lines(DATA_DIR_20K_NUM_REPL + "train.txt")
    # print(train_lines[:30])
    # print(len(train_lines))
    # start the experiments using 20k dataset with numbers replaced by @ sign
    print("skim lit")