#!/usr/bin/env python

"""
Plot the coordinates using Google Maps Static
"""
def gmplot(df, zoom=12, directory='visual/', filename='map.html'):
  ### Generate the dirs if it is not exist
  try:
    os.makedirs(directory)
  except:
    pass
  ### Google Maps Static Map generator
  from gmplot import gmplot
  lon_0 = df['longitude'].mean()
  lat_0 = df['latitude'].mean()
  lons = df['longitude'].values
  lats = df['latitude'].values
  ### Place map
  gmap = gmplot.GoogleMapPlotter(lat_0, lon_0, zoom)
  gmap.scatter(lats, lons, 'red', size=40, marker=False)
  gmap.draw('/'.join([directory, filename]))