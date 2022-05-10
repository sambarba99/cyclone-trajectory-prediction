"""
MSc Research Project: Cyclone Trajectory Forecasting

Driver code

Author: Sam Barba
"""

# Reduce TensorFlow logger spam (this is needed before any TF imports)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Python modules
from datetime import timedelta
import geopandas as gpd
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.graphics.tsaplots import plot_pacf
import sys
import tensorflow as tf
from time import perf_counter
from tqdm import tqdm

# Own modules
from data.augmentation import augment
from data.data_reader import clean_raw_data, separate_cyclone_dfs
from data.exploratory_data_analysis import do_eda, get_cyclone_duration
from graph_plotting import *
from model_builder import build_model
from model_training import train_model
from utils.constants import DATA_PATH, VERSION_NUM, FEATURES_TO_PREDICT
from utils.model_params import PARAMS

plt.rcParams['figure.figsize'] = (12, 6)   # Figure size for all plots
np.random.seed(1)                          # For reproducibility
pd.set_option('display.width', None)       # No width limit for displaying DataFrames
pd.options.mode.chained_assignment = None  # Turn off pandas error messages to avoid false positives

def print_banner(text: str) -> None:
	"""Print text within a banner"""
	text = f'  {text}  '
	hashes = '#' * 80
	print(f'\n{hashes}\n{text:#^80}\n{hashes}\n')

def main() -> None:
	"""Driver code"""

	valid = isinstance(VERSION_NUM, int) and VERSION_NUM in range(1, 5)
	assert valid, f'VERSION_NUM must be an int from 1-4 (got: {VERSION_NUM})'

	print('\nSelected model version to train/demo:', VERSION_NUM)

	# Get history_size and target_size params depending on the model
	history_size = PARAMS[VERSION_NUM]['history_size']
	target_size = PARAMS[VERSION_NUM]['target_size']

	print_banner('Python/package versions')

	print('Python version:', sys.version)
	print('TensorFlow version:', tf.__version__)
	print('Numpy version:', np.__version__)
	print('pandas version:', pd.__version__)
	print('GeoPandas version:', gpd.__version__)
	print('statsmodels version:', statsmodels.__version__)

	"""
	1. Get cleaned raw data as DataFrame format, and print its info.
	"""

	print_banner('Raw cyclone data')

	df = pd.read_csv(DATA_PATH)
	print(df)

	print_banner('Cleaned raw cyclone data')

	df = clean_raw_data()
	print(df)

	"""
	2. Separate cyclones into their own sub-dataframes according to the 'ID (name)' column.
	"""

	cyclone_dfs = separate_cyclone_dfs(df)

	"""
	3. Do exploratory data analysis on this set of dataframes.
	"""

	print_banner('Exploratory Data Analysis on raw data')

	do_eda(cyclone_dfs, all_cyclones=True)

	"""
	4. Only keep cyclone dataframes with enough recorded location history: since all model
	   versions use a certain history_size hours to predict the next target_size hours, we
	   need dataframes of cyclones that have lasted at least (history_size + target_size)
	   hours. 336 hours (2 weeks) is an arbitrary but generous selection of this value, as
	   it allows the `convert_data_to_history_and_target` function in data/train_data_prep.py
	   to generate many x and y samples to pass to the model for training.
	"""

	required_history = 336
	cyclone_dfs = [df for df in cyclone_dfs
		if get_cyclone_duration(df)[1] >= required_history]

	"""
	5. Perform augmentation of more wind speed, pressure, latitude, and longitude values in the dataframes.
	"""

	print_banner('Augmentation step')

	print(f'Selected amount of required location history per cyclone: {required_history} hrs.')
	print(f'There are {len(cyclone_dfs)} dataframes of cyclones that have a duration of at least this many hours.')
	print('The augmented versions of these will be used for training/validation/testing.')
	augmented_cyclone_dfs = []
	for df in tqdm(cyclone_dfs, desc='Augmenting each dataframe'):
		augmented_cyclone_dfs.append(augment(df))

	print_banner('Augmented data')

	print(pd.concat(augmented_cyclone_dfs))

	"""
	6. Plot some example cyclone tracks vs their augmented equivalent, and do the same
	   EDA on the new set of dataframes. Next, visualise all the cyclone tracks together.
	"""

	print_banner('Exploratory Data Analysis on augmented cyclone dataframes')

	for i in np.random.choice(len(augmented_cyclone_dfs), size=3, replace=False):
		raw_track = cyclone_dfs[i][['Latitude', 'Longitude']].values
		augmented_track = augmented_cyclone_dfs[i][['Latitude', 'Longitude']].values
		plot_raw_vs_augmented_cyclone_track(raw_track, augmented_track)

	do_eda(augmented_cyclone_dfs, all_cyclones=False)

	plot_all_cyclone_tracks(augmented_cyclone_dfs)

	"""
	7. Take some arbitrary cyclones and plot the partial autocorrelation (PACF) of
	   the main features (up to 24h of previous data).
	"""

	for i in range(3):
		lat = augmented_cyclone_dfs[i]['Latitude'].values
		long = augmented_cyclone_dfs[i]['Longitude'].values
		max_wind = augmented_cyclone_dfs[i]['Maximum Wind'].values
		min_pressure = augmented_cyclone_dfs[i]['Minimum Pressure'].values

		for (values, title) in zip([lat, long, max_wind, min_pressure],
			['latitude', 'longitude', 'maximum wind', 'minimum pressure']):

			plot_pacf(values, method='ywm', lags=24)
			plt.xlabel('Lag')
			plt.ylabel('Partial autocorrelation')
			plt.title(f'Lag values for {title} (previous 24h) (example {i + 1}/3)')
			plt.show()

	"""
	8. Train a new model or load existing saved one, and print its summary.
	"""

	print_banner('Train a model or load existing one')

	choice = input(f'Enter T to train a new v{VERSION_NUM} model,'
		f'\nor anything else to demo existing model_v{VERSION_NUM}: ').upper()

	if choice == 'T':
		model = build_model(input_timesteps=history_size, output_timesteps=target_size)
		plot_model(model, to_file=f'modelplots/model_v{VERSION_NUM}.png', show_shapes=True, show_dtype=True,
			expand_nested=True, show_layer_activations=True)
	else:
		model = load_model(f'savedmodels/model_v{VERSION_NUM}.h5')
	print(f'\nView model plot at modelplots/model_v{VERSION_NUM}.png.\n')
	model.summary()

	"""
	9. Train a new model (if option selected) on the set of augmented dataframes.
	   3 are kept unseen as a 'holdout set' for testing in section 10 below.
	"""

	# First, normalise all cyclone data
	concatenated_df = pd.concat(augmented_cyclone_dfs)
	min_values = concatenated_df[FEATURES_TO_PREDICT].min().values  # Save min/max values per feature for later
	max_values = concatenated_df[FEATURES_TO_PREDICT].max().values
	concatenated_df[FEATURES_TO_PREDICT] = concatenated_df[FEATURES_TO_PREDICT].apply(
		lambda col: (col - col.min()) / (col.max() - col.min())
	)

	# Separate them into individual dataframes again
	normalised_cyclone_dfs = separate_cyclone_dfs(concatenated_df)

	# Indices of the test set: cyclones with sufficiently different trajectories for visualisation
	test_indices = [1, 10, 20]  # 1, 15, 27
	train_cyclones = [df for idx, df in enumerate(normalised_cyclone_dfs) if idx not in test_indices]
	test_cyclones = [normalised_cyclone_dfs[i] for i in test_indices]

	if choice == 'T':
		print_banner('Model training')

		start = perf_counter()

		# No need to shuffle training data, as it's done in data/train_data_prep.py
		train_model(model, train_cyclones)

		interval = perf_counter() - start

		training_hms = str(timedelta(seconds=int(interval)))  # E.g. 3:25:45
		print(f'Completed training in {training_hms}')
		print(f'\nView loss graph at /lossgraphs/model_v{VERSION_NUM}_loss.png.')

	"""
	10. If training isn't selected, demo the saved model. Test it on 3 unseen augmented dataframes.
	"""

	print_banner(f'Testing model_v{VERSION_NUM} on unseen cyclones')

	for idx, df in enumerate(test_cyclones, start=1):
		# Visualise test cyclone track
		plot_single_cyclone_with_id(df)

		# Prepare data for model
		vals = df[FEATURES_TO_PREDICT].values
		x = vals[:history_size]                            # Let input be the first history_size hours of the cyclone
		y = vals[history_size:history_size + target_size]  # Must predict the next target_size hours

		# Apply correct input shape depending on what model is being tested
		model_input = None
		match VERSION_NUM:
			case 1: model_input = x.reshape((1, *x.shape, 1, 1))
			case 2:	model_input = x.reshape((1, *x.shape, 1))
			case 3 | 4:	model_input = x.reshape((1, *x.shape))

		# Make a prediction and 'squeeze' out unnecessary array dimensions equal to 1
		model_output = np.squeeze(model.predict(model_input))

		# model_v1 output needs formatting as follows before plotting
		if VERSION_NUM == 1:
			model_output = model_output[:, 1, :]

		# Un-normalise the data if plotting best model (v4) forecasts,
		# to get non-normalised MAE values
		if VERSION_NUM == 4:
			# Use inverse of normalisation formula (section 9)
			x = x * (max_values - min_values) + min_values
			y = y * (max_values - min_values) + min_values
			model_output = model_output * (max_values - min_values) + min_values

		# Only interested in lat/long forecast (last 2 columns)
		# We can plot model_v4 output over a world map, as it is non-normalised
		plot_cyclone_history_and_future(track_history=x[:, -2:],
			track_future=y[:, -2:],
			track_forecast=model_output[:, -2:],
			save_to_file=True,
			file_id=idx,
			plot_over_world_map=VERSION_NUM == 4)

	print('View test set output forecasts in /forecastoutputs.')

	# If a new model has been trained, ask to save it
	if choice == 'T':
		choice = input(f'\nSave newly trained model (will override model_v{VERSION_NUM} if it exists)? (Y/[N]) ').upper()
		if choice == 'Y':
			model.save(f'savedmodels/model_v{VERSION_NUM}.h5')
			print(f'\nSaved at /savedmodels/model_v{VERSION_NUM}.h5.')

if __name__ == "__main__":
	main()
