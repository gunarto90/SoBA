#!/usr/bin/env python
### General library
import numpy as np
import pandas as pd
### Utility library
import gc
### Plotting library
import matplotlib.pyplot as plt
import seaborn as sns
### Geopandas
import geopandas as gpd
import shapely
from geopandas import GeoSeries, GeoDataFrame
from shapely.geometry import Point
### Setup Directories for local library
import sys
import os
PWD = os.getcwd()
sys.path.append(PWD)
### Local library
from common.functions import *
from common.visual import *

### Global
config_file = 'config.json'
dataset_root = 'dataset/'
### Original
filename = 'checkin.csv'
### Standardized
checkin_all = 'checkin_all.csv.gz'
checkin_weekday = 'checkin_weekday.csv.gz'
checkin_weekend = 'checkin_weekend.csv.gz'
### Processed
checkin_processed_all = 'checkin_processed_all.csv.gz'
checkin_processed_weekday = 'checkin_processed_weekday.csv.gz'
checkin_processed_weekend = 'checkin_processed_weekend.csv.gz'

final_column = ['user', 'timestamp', 'latitude', 'longitude', 'location']

def generate_results(directory, df):
  ### Writing results to files
  df.to_csv('/'.join([dataset_root, directory, checkin_all]), header=True, index=False, compression ='gzip')
  df_weekday = df[df.index.dayofweek < 5]
  df_weekday.to_csv('/'.join([dataset_root, directory, checkin_weekday]), header=True, index=False, compression ='gzip')
  df_weekend = df[df.index.dayofweek >= 5]
  df_weekend.to_csv('/'.join([dataset_root, directory, checkin_weekend]), header=True, index=False, compression ='gzip')

def read_foursquare2012_checkin(write=True):
  directory = 'foursquare'
  df = pd.read_csv('/'.join([dataset_root, directory, filename]), parse_dates=['time'])
  print(df.describe(include='all'))
  print(df.head())
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
    generate_results(directory, df)

"""
directory = 'gowalla' or 'brightkite'
"""
def read_snap_stanford_checkin(directory='gowalla', write=True):
  df = pd.read_csv('/'.join([dataset_root, directory, filename]), header=None, names=['user','timestamp','latitude','longitude','location'])
  print(df.describe(include='all'))
  print(df.head())
  ### Create a datetime column as the index
  df['time'] = pd.to_datetime(df['timestamp'], unit='s')
  df = df.set_index('time')
  print(df.head())
  ### Reordering columns
  df = df[final_column]
  ### Writing results to files
  if write is True:
    generate_results(directory, df)

"""
Read the standardized data
"""
def read_processed(directory='gowalla', filename=checkin_all):
  df = pd.read_csv('/'.join([dataset_root, directory, filename]), names=final_column, header=0)
  df['u_count'] = df.groupby('user')['user'].transform('count')
  df['v_count'] = df.groupby('location')['location'].transform('count')
  ### Apply filtering
  ##  User count > 10 and location visit > 2 (otherwise there is no co-location)
  df = df[(df['u_count'] > 10) & (df['v_count'] > 1)]
  df.drop(['u_count', 'v_count'], axis=1)
  # print(df.describe(include='all'))
  # print(df.head(10))

  return df

def preprocess_data():
  write = False
  # read_foursquare2012_checkin(write)
  # read_snap_stanford_checkin('brightkite', write)
  # read_snap_stanford_checkin('gowalla', write)

def visualize_data(df):
  shapely.speedups.enable()
  geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
  temp = df.drop(['longitude', 'latitude'], axis=1)
  crs = {'init': 'epsg:4326'}
  gdf = GeoDataFrame(temp, crs=crs, geometry=geometry)

  print(gdf)
  gdf.plot(marker='*', color='green', markersize=50, figsize=(5, 5));
  plt.show()

  del temp
  del gdf
  gc.collect()

def main():
  ### Read config
  config = read_config()
  ### Read original data and generate standardized data
  # preprocess_data()

  ### Read standardized data and perform preprocessing
  df = read_processed('foursquare')
  df = df[0:10000]  ### For testing purpose --> to speed-up and understand the data
  gmplot(df)
  ### Visualize the dataframe
  # visualize_data(df)

if __name__ == '__main__':
  main()