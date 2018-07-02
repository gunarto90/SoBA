#!/usr/bin/env python
# coding=utf-8
import os
import shutil
import json
import sys
from datetime import datetime, date
import time
from functools import wraps
from math import radians, cos, sin, asin, sqrt, pow, exp, log
import numpy as np
import pandas as pd

IS_DEBUG = True
LOG_PATH = 'log'

"""
Configurations
"""
def read_config(filename='config.json'):
  try:
    with open(filename) as data_file:
      config = json.load(data_file)
      return config
  except IOError as ex:
    print('File not found : %s' % filename)
    print('Please create a file named \"config.json\" in the root directory. The contents of the file should refer to the file \"config.json.example\"')
  except Exception as ex:
    print('Exception in init config file : %s' % ex)
  return None

def init_all_folders(config):
  pass

"""
Logging tools
"""
def error(message, source, out_dir=LOG_PATH, out_file='error.txt'):
  try:
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
  except Exception as ex:
    if IS_DEBUG:
      print(ex)
  try:
    with open('/'.join([out_dir, out_file]), 'a') as fw:
      fw.write('[{}] ({}) {}\n'.format(datetime.now(), source, message))
  except Exception as ex:
    if IS_DEBUG:
      print(ex)

def debug(*argv):
  if not IS_DEBUG:
    return
  if len(argv) == 0 or argv is None:
    return
  try:
    # s = ' '.join(map(str, argv))
    s = ' '.join(map(lambda s: unicode(str(s), 'utf-8'), argv))
    # s = unicode(str(s), 'utf-8')
    print('[{}] {}'.format(datetime.now(), s))
    try:
      log_name = '.'.join([str(date.today()), 'txt'])
      log_path = '/'.join([LOG_PATH, log_name])
      with open(log_path, 'a') as fw:
        fw.write('[{}] {}\n'.format(str(datetime.now()), s))
    except Exception as ex:
      if IS_DEBUG:
        print(ex)
    sys.stdout.flush()
  except Exception as ex:
    error(str(ex), source='lalala.functions.py/debug')

def report_progress(counter, start, finish, context='', every_n=100):
  if counter % every_n == 0:
    debug('Processing {} of {} {} ({:.3f}%) '.format(
      counter, (finish-start), context, float(counter)*100.0/(finish-start)),
    )
 
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        debug("Total time running \"%s\": %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer

### Metrics
"""
Input: array of float
Output: float
"""
def entropy(data):
  total = 0.0
  ent = 0
  for item in data:
    total += item
  for item in data:
    pi = float(item)/total
    ent -= pi * log(pi)
  return ent

def entropy_np(data):
  total = sum(data)
  if total <= 0:
    return 0.0
  p_i = data / total
  p_i = np.trim_zeros(p_i)
  p_i = -1 * p_i * np.log(p_i)
  return sum(p_i)

"""
Input: shapely points
Output: float
"""
def earth_distance(point1, point2):
  return haversine(point1.x, point1.y, point2.x, point2.y)

"""
Input: float
Output: float
"""
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
  return distance # in meter

"""
Input: array of float
Output: array of float
"""
def haversine_np(lon1, lat1, lon2, lat2):
  """
  Calculate the great circle distance between two points
  on the earth (specified in decimal degrees)
  All args must be of equal length (array-like)
  """
  lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

  dlon = lon2 - lon1
  dlat = lat2 - lat1

  a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

  c = 2 * np.arcsin(np.sqrt(a))
  km = 6367 * c
  distance = km * 1000
  return distance # in meter

"""
IO tools
"""
def make_sure_path_exists(path):
  try:
    os.makedirs(path)
    return True
  except OSError:
    return False

def is_file_exists(filename):
  return os.path.isfile(filename)
  # try:
  #   with open(filename, 'r'):
  #     return True
  # except:
  #   return False

def remove_file_if_exists(filename):
  try:
    os.remove(filename)
  except OSError:
    pass

def remove_all_files(folder, remove_dir=False):
  for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
          os.unlink(file_path)
        elif os.path.isdir(file_path) and remove_dir is True: 
          shutil.rmtree(file_path)
    except Exception as e:
        print(e)

def init_begin_end(n_core, size, start=0, finish=-1):
  begin = []
  end = []
  n_chunks = 50
  iteration = n_core*n_chunks
  size_per_chunk = int(size / iteration)
  for i in range(iteration):
    if i == 0:
      begin.append(0)
    else:
      begin.append(i*size_per_chunk)
    if i == iteration - 1:
      end.append(size)
    else:
      end.append((i+1)*size_per_chunk)
  ### If the start and finish are different from default
  if start < 0:
    start = 0
  if finish > size:
    finish = size
  if start == 0 and finish == -1:
    pass
  elif start == 0 and finish == 0:
    del begin[:], end[:]
  else:
    if finish == -1:
      finish = size
    idx_start = -1
    idx_finish = -1
    for i in range(len(begin)):
      if begin[i] >= start:
        idx_start = i
        break
    for i in xrange(len(end)-1, -1, -1):
      if finish >= end[i]:
        idx_finish = i+1
        break
    if idx_start == idx_finish and idx_finish < len(end)-1:
      idx_finish += 1
    begin = begin[idx_start:idx_finish]
    end = end[idx_start:idx_finish]
  assert len(begin) == len(end) ### Make sure the length of begin == length of end
  return begin, end

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result