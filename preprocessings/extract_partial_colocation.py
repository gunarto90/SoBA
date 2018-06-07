#!/usr/bin/env python
import numpy as np
import pandas as pd
import os, sys
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Custom libraries
from common.functions import IS_DEBUG, read_config, debug, fn_timer
from preprocessings.read import extract_checkins_per_user, extract_checkins_per_venue, extract_checkins_all
from methods.colocation import process_map, process_reduce, prepare_colocation

def extract_colocation(config, p, k, t_diff_input, s_diff_input, t_diff_target, s_diff_target):
  ### Format: user1,user2,location1,location2,time1,time2,lat1,lon1,lat2,lon2,t_diff,s_diff
  working_directory = config['directory']['colocation']
  filename  = config['intermediate']['colocation']
  debug('Reading colocation file', '/'.join([working_directory, filename.format(p, k, t_diff_input, s_diff_input)]))
  df = pd.read_csv('/'.join([working_directory, filename.format(p, k, t_diff_input, s_diff_input)]))
  debug('Original colocation size', len(df))
  df = df[(df['t_diff'] <= t_diff_target) & (df['s_diff'] <= s_diff_target)]
  debug('Filtered colocation size', len(df))
  debug('Writing colocation file', '/'.join([working_directory, filename.format(p, k, t_diff_target, s_diff_target)]))
  df.to_csv('/'.join([working_directory, filename.format(p, k, t_diff_target, s_diff_target)]), index=False)

def main():
  debug('Started extracting partial colocation')
  ### Read config
  config = read_config()
  config_partial = config['kwargs']['partial_colocation']
  p = config_partial['p']
  k = config_partial['k']
  t_input = config_partial['t_input']
  d_input = config_partial['d_input']
  t_target = config_partial['t_target']
  d_target = config_partial['d_target']
  extract_colocation(config, p, k, t_input, d_input, t_target, d_target)
  debug('Finished extracting partial colocation')

if __name__ == '__main__':
  main()