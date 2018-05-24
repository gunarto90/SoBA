#!/usr/bin/env python
"""
Please run this code under the root directory using the command 'python preprocessing/read.py'
"""
### General library
import numpy as np
import pandas as pd
### Utility library
import pickle
import sys
import os.path
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local library
from common.functions import IS_DEBUG, debug, fn_timer, make_sure_path_exists, is_file_exists, read_config
from common.visual import gmplot
from methods.colocation import generate_colocation

RAW_CHECKIN_FILE = 'checkin.csv'

### Global
config_file = 'config.json'
final_column = ['user', 'timestamp', 'latitude', 'longitude', 'location']

"""
Generate three standardized compressed csv files which consist of all, weekend, and weekday data
"""
@fn_timer
def generate_results(root, dataset, df):
  ### Standardized
  checkin_all = 'checkin_all.csv.gz'
  checkin_weekday = 'checkin_weekday.csv.gz'
  checkin_weekend = 'checkin_weekend.csv.gz'
  ### Writing results to files
  df.to_csv('/'.join([root, dataset, checkin_all]), header=True, index=False, compression ='gzip')
  df_weekday = df[df.index.dayofweek < 5]
  df_weekday.to_csv('/'.join([root, dataset, checkin_weekday]), header=True, index=False, compression ='gzip')
  df_weekend = df[df.index.dayofweek >= 5]
  df_weekend.to_csv('/'.join([root, dataset, checkin_weekend]), header=True, index=False, compression ='gzip')

"""
Reading the foursquare checkin dataset
"""
@fn_timer
def read_foursquare2012_checkin(root, write=True):
  dataset = 'foursquare'
  debug('Read Checkin %s' %dataset)
  df = pd.read_csv('/'.join([root, dataset, RAW_CHECKIN_FILE]), parse_dates=['time'])
  debug(df.describe(include='all'))
  debug(df.head())
  ### Create a UNIX timestamp column from the datetime format
  df['timestamp'] = df['time'].values.astype(np.int64) // 10 ** 9
  ### Set the datetime as the index
  df = df.set_index('time')
  ### Reordering columns
  df = df[final_column]
  ### Error checking
  # odd = df.loc[df.longitude>-80, ['longitude', 'latitude']]
  ### Writing results to files
  if write is True:
    generate_results(root, dataset, df)

"""
Reading the checkin dataset from SNAP Stanford
dataset = 'gowalla' or 'brightkite'
"""
@fn_timer
def read_snap_stanford_checkin(root, dataset='gowalla', write=True):
  debug('Read SNAP Stanford Checkin %s' %dataset)
  df = pd.read_csv('/'.join([root, dataset, RAW_CHECKIN_FILE]), header=None, names=['user','timestamp','latitude','longitude','location'])
  debug(df.describe(include='all'))
  debug(df.head())
  ### Create a datetime column as the index
  df['time'] = pd.to_datetime(df['timestamp'], unit='s')
  df = df.set_index('time')
  debug(df.head())
  ### Reordering columns
  df = df[final_column]
  ### Writing results to files
  if write is True:
    generate_results(root, dataset, df)

"""
Read the standardized data
"""
@fn_timer
def read_processed(root, dataset='gowalla', mode='all'):
  filename = 'checkin_{}.csv.gz'.format(mode)
  df = pd.read_csv('/'.join([root, dataset, filename]), names=final_column, header=0)
  df['u_count'] = df.groupby('user')['user'].transform('count')
  df['v_count'] = df.groupby('location')['location'].transform('count')
  ### Apply filtering
  ##  User count > 10 and location visit > 2 (otherwise there is no co-location)
  df = df[(df['u_count'] > 10) & (df['v_count'] > 1)]
  df.drop(['u_count', 'v_count'], axis=1, inplace=True)

  return df

def preprocess_data(root):
  write = False
  read_foursquare2012_checkin(root, write)
  read_snap_stanford_checkin(root, 'brightkite', write)
  read_snap_stanford_checkin(root, 'gowalla', write)

def visualize_data(df):
  test_limit = 100
  temp = df[0:test_limit]  ### For testing purpose --> to speed-up and understand the data
  gmplot(temp)

"""
Extract all the checkins and group them on each user
Input:
- dataset_name (foursquare, gowalla, brightkite)
- mode (all, weekday, weekend)
- config: config.json filename
Output:
- Dictionary (uid, checkins) (int, dataframe)
"""
@fn_timer
def extract_checkins_per_user(dataset_name, mode, config):
  dataset_root = config['directory']['dataset']
  intermediate_root = config['directory']['intermediate']
  checkins_intermediate_file = config['intermediate']['checkins']
  debug('Processing %s [%s]' % (dataset_name, mode))
  pickle_directory = '/'.join([intermediate_root, dataset_name])
  make_sure_path_exists(pickle_directory)
  pickle_filename = '/'.join([pickle_directory, checkins_intermediate_file.format(mode)])
  if not is_file_exists(pickle_filename):
    df = read_processed(dataset_root, dataset_name, mode)
    debug('#checkins', len(df))
    uids = df['user'].unique()
    checkins_per_user = {}
    for uid in uids:
      checkins_per_user[uid] = df.loc[df['user'] == uid]  ### Need to order by "timestamp"
    with open(pickle_filename, 'wb') as handle:
      pickle.dump(checkins_per_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
  else:
    with open(pickle_filename, 'rb') as handle:
      checkins_per_user = pickle.load(handle)
  debug('#users', len(checkins_per_user))
  return checkins_per_user

@fn_timer
def main():
  ### Read config
  config = read_config()

  ### Read original data and generate standardized data
  # dataset_root = config['directory']['dataset']
  # preprocess_data(dataset_root)

  ### Read standardized data and perform preprocessing
  datasets = config['active_dataset']
  modes = config['active_mode']
  for dataset_name in datasets:
    for mode in modes:
      checkins_per_user = extract_checkins_per_user(dataset_name, mode, config)
      debug(len(checkins_per_user))

  # geometry = pd.DataFrame(df, columns=['latitude', 'longitude', 'timestamp']).values
  # debug(geometry.shape, geometry)
  # geometry = np.reshape(geometry, (-1, 2))
  # tree = KDTree(geometry)
  # debug(geometry[0])
  # dist, ind = tree.query(np.array(geometry[0]).reshape(-1,2), k=3)
  # debug(ind, dist)

  ### Convert the pandas dataframe to geopandas dataframe
  # gdf = convert_to_geopandas(df)

  ### Visualize the dataframe
  # visualize_data(df)

if __name__ == '__main__':
  main()