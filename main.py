#!/usr/bin/env python
from joblib import Parallel, delayed
import math
### Custom libraries
from common.functions import IS_DEBUG, read_config, debug, fn_timer
from preprocessings.read import extract_checkins_per_user
from methods.colocation import process_map, process_reduce, prepare_colocation

def init_begin_end(n_core, size):
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
  assert len(begin) == len(end) ### Make sure the length of begin == length of end
  return begin, end

@fn_timer
def map_reduce_colocation(config, checkins, p, k, t_diff, s_diff):
  n_core = config['n_core']
  ### For the sake of parallelization
  begins, ends = init_begin_end(n_core, len(checkins))
  # debug('Begins', begins)
  # debug('Ends', ends)
  ### Generate colocation based on extracted checkins
  prepare_colocation(config, p, k, t_diff, s_diff, begins, ends)
  Parallel(n_jobs=n_core)(delayed(process_map)(checkins, config, begins[i], ends[i], p, k, t_diff, s_diff) for i in range(len(begins)))
  process_reduce(config, p, k, t_diff, s_diff)
  debug('Finished map-reduce for [p%d, k%d, t%d, d%d]' % (p, k, t_diff, s_diff))

def run_colocation(config):
  ### Read standardized data and perform preprocessing
  n_core = config['n_core']
  all_datasets = config['dataset']
  all_modes = config['mode']
  datasets = config['active_dataset']
  modes = config['active_mode']
  t_diffs = config['ts']
  s_diffs = config['ds']
  for dataset_name in datasets:
    p = all_datasets.index(dataset_name)
    for mode in modes:
      k = all_modes.index(mode)
      debug('Dataset', dataset_name, p, 'Mode', mode, k, '#Core', n_core)
      ### Extracting checkins for each user
      checkins = extract_checkins_per_user(dataset_name, mode, config)
      for t_diff in t_diffs:
        for s_diff in s_diffs:
          map_reduce_colocation(config, checkins, p, k, t_diff, s_diff)

def main():
  ### Read config
  config = read_config()
  ### Co-location generation
  run_colocation(config)

if __name__ == '__main__':
  main()