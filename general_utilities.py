import os
from datetime import datetime
from pympler import asizeof

IS_DEBUG = True

def show_object_size(obj, name):
    size = asizeof.asizeof(obj)
    print('Size of {0} is : {1:,} Bytes'.format(name, size))

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        pass

def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def write_to_file(filename, text, append=True, add_linefeed=True):
    if append:
        mode = 'a'
    else:
        mode = 'w'
    linefeed = ''
    if add_linefeed is True:
        linefeed = '\n'
    with open(filename, mode) as fw:
        fw.write(str(text) + linefeed)
    pass

def write_to_file_buffered(filename, text_list, append=True):
    buffer_size = 10000
    counter = 0
    temp_str = ""
    for text in text_list:
        if counter <= buffer_size:
            temp_str = temp_str + text + '\n'
        else:
            write_to_file(filename, temp_str, append, add_linefeed=False)
            temp_str = ""
            counter = 0
        counter += 1
    # Write remaining text
    if temp_str != "":
        write_to_file(filename, temp_str, append, add_linefeed=False)

def debug(message, callerid=None):
    if IS_DEBUG == False:
        return
    if callerid is None:
        print('[DEBUG] [{1}] {0}'.format(message, datetime.now()))
    else :
        print('[DEBUG] [{2}] <Caller: {1}> {0}'.format(message, callerid, datetime.now()))