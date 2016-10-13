import os
import math
from datetime import datetime
from datetime import date
from pympler import asizeof
from math import radians, cos, sin, asin, sqrt, pow, exp

IS_DEBUG = True

def show_object_size(obj, name):
    size = asizeof.asizeof(obj)
    print('Size of {0} is : {1:,} Bytes'.format(name, size))

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
        return True
    except OSError as exception:
        return False

def is_file_exists(filename):
    try:
        with open(filename, 'r'):
            return True
    except:
        return False

def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def write_to_file(filename, text, append=True, add_linefeed=True):
    if append is True:
        mode = 'a'
    else:
        mode = 'w'
    linefeed = ''
    if add_linefeed is True:
        linefeed = '\n'
    with open(filename, mode) as fw:
        fw.write(str(text) + linefeed)

def write_to_file_buffered(filename, text_list, append=True):
    debug('Writing file: {}'.format(filename))
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

def debug(message, callerid=None, clean=False, out_stdio=True, out_file=True):
    make_sure_path_exists('./log')
    debug_filename = 'log/log_{}.txt'.format(date.today())
    if IS_DEBUG == False:
        return
    text = ''
    if clean is False:
        if callerid is None:
            text = '[DEBUG] [{1}] {0}'.format(message, datetime.now())
        else :
            text = '[DEBUG] [{2}] <Caller: {1}> {0}'.format(message, callerid, datetime.now())
    else:
        if callerid is None:
            text = message
        else :
            text = '{0} <Caller: {1}>'.format(message, callerid)
    if out_stdio is True:
        print(text)
    if out_file is True:
        write_to_file(debug_filename, text)
    return text

def entropy(data):
    total = 0.0
    ent = 0
    for item in data:
        total += item
    for item in data:
        pi = float(item)/total
        ent -= pi * math.log(pi)
    return ent

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    distance = km * 1000
    return distance