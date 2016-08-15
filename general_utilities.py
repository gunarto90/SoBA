import os
from pympler import asizeof

def show_object_size(obj, name):
    size = asizeof.asizeof(obj)
    print('Size of {0} is : {1:,} Bytes'.format(name, size))

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        pass

def write_to_file(filename, text, append=True):
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(filename, mode) as fw:
        fw.write(str(text) + '\n')
    pass