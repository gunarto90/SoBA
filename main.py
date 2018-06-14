#!/usr/bin/env python
from joblib import Parallel, delayed
import math
### Custom libraries
from common.functions import IS_DEBUG, read_config, debug, fn_timer
from preprocessings.read import extract_checkins_per_user, extract_checkins_per_venue, extract_checkins_all
from methods.colocation import process_map, process_reduce, prepare_colocation
from methods.sci import extract_popularity, extract_colocation_features

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

@fn_timer
def map_reduce_colocation(config, checkins, p, k, t_diff, s_diff):
  kwargs = config['kwargs']
  n_core = kwargs['n_core']
  start = kwargs['colocation']['start']
  finish = kwargs['colocation']['finish']
  ### For the sake of parallelization
  begins, ends = init_begin_end(n_core, len(checkins), start=start, finish=finish)
  debug('Begins', begins)
  debug('Ends', ends)
  ### Generate colocation based on extracted checkins
  prepare_colocation(config, p, k, t_diff, s_diff, begins, ends)
  Parallel(n_jobs=n_core)(delayed(process_map)(checkins, config, begins[i-1], ends[i-1], p, k, t_diff, s_diff) for i in xrange(len(begins), 0, -1))
  process_reduce(config, p, k, t_diff, s_diff)
  debug('Finished map-reduce for [p%d, k%d, t%d, d%.3f]' % (p, k, t_diff, s_diff))

def extract_checkins(config, dataset_name, mode, run_by):
  ### Extracting checkins
  if run_by == 'venue': ### If extracted by each venue (Simplified SIGMOD 2013 version)
    checkins = extract_checkins_per_venue(dataset_name, mode, config)
  elif run_by == 'checkin': ### Map-reduce fashion but per check-in
    checkins = extract_checkins_all(dataset_name, mode, config)
  else: ### Default is by each user
    checkins = extract_checkins_per_user(dataset_name, mode, config)
  return checkins

def run_colocation(config, run_by='user'):
  ### Read standardized data and perform preprocessing
  kwargs = config['kwargs']
  all_datasets = config['dataset']
  all_modes = config['mode']
  n_core = kwargs['n_core']
  datasets = kwargs['active_dataset']
  modes = kwargs['active_mode']
  t_diffs = kwargs['ts']
  s_diffs = kwargs['ds']
  for dataset_name in datasets:
    p = all_datasets.index(dataset_name)
    for mode in modes:
      k = all_modes.index(mode)
      debug('Run co-location on Dataset', dataset_name, p, 'Mode', mode, k, '#Core', n_core)
      ### Extracting checkins
      checkins = extract_checkins(config, dataset_name, mode, run_by)
      for t_diff in t_diffs:
        for s_diff in s_diffs:
          map_reduce_colocation(config, checkins, p, k, t_diff, s_diff)
      checkins.clear()
      del checkins

def run_sci(config):
  ### Read standardized data and perform preprocessing
  kwargs = config['kwargs']
  all_datasets = config['dataset']
  all_modes = config['mode']
  n_core = kwargs['n_core']
  datasets = kwargs['active_dataset']
  modes = kwargs['active_mode']
  t_diffs = kwargs['ts']
  s_diffs = kwargs['ds']
  for dataset_name in datasets:
    p = all_datasets.index(dataset_name)
    for mode in modes:
      k = all_modes.index(mode)
      debug('Run SCI on Dataset', dataset_name, p, 'Mode', mode, k, '#Core', n_core)
      ### Extracting checkins
      checkins = extract_checkins(config, dataset_name, mode, 'user')
      stat_lp = extract_popularity(checkins, config, p, k)
      for t_diff in t_diffs:
        for s_diff in s_diffs:
          extract_colocation_features(stat_lp, config, p, k, t_diff, s_diff)

def main():
  ### Started the program
  debug('Started SCI+')
  ### Read config
  config = read_config()
  kwargs = config['kwargs']
  ### Co-location
  is_run_colocation = kwargs['colocation']['run']
  run_by = kwargs['colocation']['run_by']
  if is_run_colocation is not None and is_run_colocation is True:
    ### Co-location generation
    run_colocation(config, run_by)
  ### SCI
  is_run_sci = kwargs['sci']['run']
  if is_run_sci is not None and is_run_sci is True:
    run_sci(config)
  ### Finished the program
  debug('Finished SCI+')

if __name__ == '__main__':
  main()