import pickle

def load_pickle(path, mode = "rb"):
    with open(path, mode) as f:
        data = pickle.load(f)
    return data

def dump_pickle(path, data, mode = "wb"):
    with open(path, mode):
        pickle.dump(data, path)