import numpy as np


def split_csv_by_lines(csv_path, dests, p, header=True):
    """Split a csv file by lines.

    Example:
        split_csv_by_lines('train.csv', dests=['data/train.csv', 'data/val.csv'], p=[0.8,0.2])

    Arguments:
        csv_path {[type]} -- [description]
        p {list} -- a list of sampling probabilities

    Keyword Arguments:
        dests {[type]} -- [description] (default: {None})
        header {bool} -- [description] (default: {True})
    """

    assert len(dests) == len(p)

    with open(csv_path) as csv:
        files = [open(dest, "w") for dest in dests]

        if header:
            header = csv.readline()
            for file in files:
                file.write(header)

        for line in csv:
            out = np.random.choice(files, p=p)
            out.write(line)

        for file in files:
            file.close()


# split_csv_by_lines('/data/whale/train.csv', dests=['0.csv', '1.csv', '2.csv'], p=[0.8, 0.1, 0.1])
