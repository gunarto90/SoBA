#!/usr/bin/env python
from common.functions import IS_DEBUG, read_config, debug
from preprocessings.read import extract_checkins_per_user
from methods.colocation import generate_colocation

def init_begin_end(n_core, arr_size):
  begin = []
  end = []
  for i in range(n_core):
    if i == 0:
      begin.append(0)
    else:
      begin.append(1+int(arr_size/n_core)*(i))
    if i == n_core - 1:
      end.append(arr_size)
    else:
      end.append(int(arr_size/n_core)*(i+1))

  return begin, end

def run_colocation(config):
  ### Read standardized data and perform preprocessing
  n_core = config['n_core']
  datasets = config['active_dataset']
  modes = config['active_mode']
  for dataset_name in datasets:
    for mode in modes:
      debug('Dataset', dataset_name, 'Mode', mode, '#Core', n_core)
      ### Extracting checkins for each user
      checkins_per_user = extract_checkins_per_user(dataset_name, mode, config)
      ### For the sake of parallelization
      begin, end = init_begin_end(n_core, len(checkins_per_user))
      debug(begin)
      debug(end)
      
      ### Generate colocation based on extracted checkins
      # colocations = generate_colocation(checkins_per_user, start=0, finish=10)  ### Start should be 0 and finish should be len(checkins_per_user)
      # debug(colocations)

def main():
  ### Read config
  config = read_config()
  ### Co-location generation
  run_colocation(config)

if __name__ == '__main__':
  main()