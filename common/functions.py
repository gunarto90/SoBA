#!/usr/bin/env python
# coding=utf-8
import os
import json
import sys
from datetime import datetime

"""
Configurations
"""
def read_config(filename='config.json'):
  try:
    with open(filename) as data_file:
      config = json.load(data_file)
      return config
  except Exception as ex:
    print('Exception in init config file : %s' % ex)
  return None

"""
Logging tools
"""
def error(message, source, out_dir='log', out_file='error.txt'):
  try:
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
  except Exception as ex:
    print(ex)
  try:
    with open('/'.join([out_dir, out_file]), 'a') as fw:
      fw.write('[{}] ({}) {}\n'.format(datetime.now(), source, message))
  except Exception as ex:
    print(ex)

def debug(*argv):
  if len(argv) == 0 or argv is None:
    return
  try:
    # s = ' '.join(map(str, argv))
    s = ' '.join(map(lambda s: unicode(str(s), 'utf-8'), argv))
    # s = unicode(str(s), 'utf-8')
    print('[{}] {}'.format(datetime.now(), s))
    sys.stdout.flush()
  except Exception as ex:
    error(str(ex), source='lalala.functions.py/debug')