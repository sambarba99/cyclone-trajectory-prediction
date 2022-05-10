"""
MSc Research Project: Cyclone Trajectory Forecasting

Constants

Author: Sam Barba
"""

import geopandas as gpd

# Change this to whichever model version (1 - 4) you'd like to train/demo
VERSION_NUM = 4

DATA_PATH = '#cyclonedata\\data.csv'

COLS_TO_KEEP = ['ID', 'Name', 'Date', 'Time', 'Latitude', 'Longitude', 'Maximum Wind', 'Minimum Pressure']

# Predict these 4, but only interested in lat/long forecasts
# (feed all 4 to model however, as feeding only lat/long values
# during training would result in worse accuracy - see report section 3.1)
FEATURES_TO_PREDICT = ['Maximum Wind', 'Minimum Pressure', 'Latitude', 'Longitude']

EARTH_RADIUS = 6371  # Kilometres

CUSTOM_DATETIME_FORMAT = '%d.%m.%Y %H:%M'

PANDAS_DEFAULT_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# Proportion of training set size
# (means that proportion of validation set = 0.2)
TRAIN_PROP = 0.8

# For plotting cyclone trajectories on top of
WORLD_MAP = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
