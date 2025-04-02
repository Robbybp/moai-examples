import os


FILEDIR = os.path.dirname(__file__)


def _validate_dir(path, create):
    if os.path.exists(path) and not os.path.isdir(path):
        raise OSError("{path} exists and is not a directory")
    elif create and not os.path.exists(path):
        os.mkdir(path)
        return path
    else:
        return path


def get_data_dir(create=False):
    datadir = os.path.join(FILEDIR, "data")
    return _validate_dir(datadir, create)


def get_nn_dir(create=False):
    nndir = os.path.join(FILEDIR, "nn-models")
    return _validate_dir(nndir, create)
