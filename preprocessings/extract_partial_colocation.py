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

def extract_colocation(config, p, k, t_diff_input, s_diff_input, t_diff_targets, s_diff_targets):
  ### Format: user1,user2,location1,location2,time1,time2,lat1,lon1,lat2,lon2,t_diff,s_diff
  working_directory = config['directory']['colocation']
  in_filename  = config['intermediate']['colocation']
  out_filename = config['intermediate']['colocation_compressed']
  debug('Reading colocation file', '/'.join([working_directory, in_filename.format(p, k, t_diff_input, s_diff_input)]))
  df = pd.read_csv('/'.join([working_directory, in_filename.format(p, k, t_diff_input, s_diff_input)]))
  debug('Original colocation size', len(df))
  debug('t_diff', df['t_diff'].max(), 's_diff', df['s_diff'].max())
  for t_diff_target in t_diff_targets:
    df = df[(df['t_diff'] <= t_diff_target)]
    for s_diff_target in s_diff_targets:
      output_final_name = '/'.join([working_directory, out_filename.format(p, k, t_diff_target, s_diff_target)])
      df_temp = df[(df['s_diff'] <= s_diff_target)]
      debug('Filtered colocation size', len(df_temp))
      debug('t_diff', df_temp['t_diff'].max(), 's_diff', df_temp['s_diff'].max())
      debug('Writing colocation file', output_final_name)
      df_temp.to_csv(output_final_name, index=False, compression='bz2')
      del df_temp

def main():
  debug('Started extracting partial colocation')
  ### Read config
  config = read_config()
  config_partial = config['kwargs']['partial_colocation']
  p = config_partial['p']
  k = config_partial['k']
  t_input = config_partial['t_input']
  d_input = config_partial['d_input']
  t_targets = config_partial['t_target']
  d_targets = config_partial['d_target']
  extract_colocation(config, p, k, t_input, d_input, t_targets, d_targets)
  debug('Finished extracting partial colocation')

if __name__ == '__main__':
  main()