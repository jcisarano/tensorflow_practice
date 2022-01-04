

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
                line_data["total_lines"] = len(abstract_line_split) - 1
                abstract_samples.append(line_data)
        else:
            abstract_lines += line

    return abstract_samples


def get_labels_one_hot(labels):
    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot = one_hot_encoder.fit_transform(labels)

    return one_hot


def get_labels_int_encoded(labels):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    return labels_encoded, label_encoder.classes_