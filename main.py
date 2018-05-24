#!/usr/bin/env python
import csv
import io
from common.functions import IS_DEBUG, read_config, debug, make_sure_path_exists
from preprocessings.read import extract_checkins_per_user
from methods.colocation import generate_colocation, extract_geometry

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

def write_colocation(data, config, p, k, t, d):
  directory = config['directory']['colocation']
  filename  = config['intermediate']['colocation']
  make_sure_path_exists(directory)
  output = io.BytesIO()
  writer = csv.writer(output)
  for row in data:
    writer.writerow(row)
  with open('/'.join([directory, filename.format(p,k,t,d)]), 'wb') as f:
    f.write('user1,user2,location1,location2,time1,time2,lat1,lon1,lat2,lon2,t_diff,s_diff\n')
    f.write(output.getvalue())

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
      checkins_per_user = extract_checkins_per_user(dataset_name, mode, config)
      ### For the sake of parallelization
      # begin, end = init_begin_end(n_core, len(checkins_per_user))
      # debug(begin)
      # debug(end)
      for t_diff in t_diffs:
        for s_diff in s_diffs:
          ### Generate colocation based on extracted checkins
          colocations = generate_colocation(checkins_per_user, start=0, finish=10, t_diff=t_diff, s_diff=s_diff)  ### Start should be 0 and finish should be len(checkins_per_user)
          write_colocation(colocations, config, p, k, t_diff, s_diff)

def main():
  ### Read config
  config = read_config()
  ### Co-location generation
  run_colocation(config)

if __name__ == '__main__':
  main()