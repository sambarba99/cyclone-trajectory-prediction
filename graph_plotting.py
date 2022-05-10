"""
MSc Research Project: Cyclone Trajectory Forecasting

Graph plotting functionality

Author: Sam Barba
"""

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from tensorflow.keras.callbacks import History

from utils.constants import VERSION_NUM, WORLD_MAP
from utils.error_calc import mae

def plot_world_map(*, min_lat: float = -90, max_lat: float = 90,
	min_long: float = -180, max_long: float = 180) -> None:

	"""Plot world map for a cyclone trajectory to be plotted on top of"""

	WORLD_MAP.plot(color='#b0e8b0', edgecolor='#609060', linewidth=0.5)
	plt.xlim([min_long, max_long])
	plt.ylim([min_lat, max_lat])

def plot_raw_vs_augmented_cyclone_track(raw_track: np.ndarray, augmented_track: np.ndarray) -> None:
	"""Plot an un-augmented cyclone track vs its augmented equivalent"""

	lat_raw, long_raw = raw_track.T
	lat_augmented, long_augmented = augmented_track.T
	min_lat, max_lat = min(lat_augmented), max(lat_augmented)
	min_long, max_long = min(long_augmented), max(long_augmented)

	# Plot world map then cyclone track on top
	plot_world_map(min_lat=min_lat - 2, max_lat=max_lat + 2, min_long=min_long - 2, max_long=max_long + 2)

	plt.plot(long_raw, lat_raw, color='red', linewidth=1, zorder=1)
	plt.scatter(long_raw, lat_raw, color='red', label='Original track data', zorder=1)
	plt.scatter(long_augmented, lat_augmented,
		color='#0080ff', s=20, alpha=0.5, marker='^',
		label='Augmented data', zorder=2)

	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.legend()
	plt.title('Example of raw cyclone track vs track with augmented data')
	plt.show()

def plot_attribute_bar_graph_multi_cyclones(*, cyclone_attributes: list,
	measured_quantity: str, unit: str, for_all_cyclones: bool) -> None:

	"""
	Plot a bar graph showing the distribution of a certain attribute across many cyclones
	(e.g. duration, displacement, speed, max wind, min pressure) for EDA
	"""

	# Highest attribute values towards the right side of the graph
	cyclone_attributes = sorted(cyclone_attributes)

	plt.bar(range(len(cyclone_attributes)), height=cyclone_attributes, width=1, color='#0080ff')
	plt.xlabel('Cyclone')
	plt.ylabel(f'{measured_quantity.capitalize()} ({unit})')
	if for_all_cyclones:
		plt.title(f'Measurements of {measured_quantity} for all ({len(cyclone_attributes)}) cyclones'
			'\n(cyclones with missing values excluded)')
	else:
		plt.title(f'Measurements of {measured_quantity} for augmented ({len(cyclone_attributes)}) cyclones')
	plt.show()

def plot_all_cyclone_tracks(cyclone_dfs: list) -> None:
	"""Plot all cyclone tracks that will be used for training/validation/testing"""

	lat_lists = [list(df['Latitude'].values) for df in cyclone_dfs]
	long_lists = [list(df['Longitude'].values) for df in cyclone_dfs]
	lat_lists = sum(lat_lists, start=[])  # Flatten lists into 1
	long_lists = sum(long_lists, start=[])
	min_lat, max_lat = min(lat_lists), max(lat_lists)
	min_long, max_long = min(long_lists), max(long_lists)

	plot_world_map(min_lat=min_lat - 2, max_lat=max_lat + 2, min_long=min_long - 2, max_long=max_long + 2)

	for df in cyclone_dfs:
		lat = df['Latitude'].values
		long = df['Longitude'].values
		plt.plot(long, lat, color='#0080ff', linewidth=2, alpha=0.5)

	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.title(f'All {len(cyclone_dfs)} cyclone tracks used for training/validation/testing')
	plt.show()

def plot_single_cyclone_with_id(cyclone_df: DataFrame) -> None:
	"""Plot a single cyclone trajectory with the cyclone ID, for testing section"""

	cyclone_id = cyclone_df['ID (name)'].values[0]
	lat = cyclone_df['Latitude'].values
	long = cyclone_df['Longitude'].values

	plt.plot(long, lat, color='#0080ff', zorder=1)
	plt.scatter(long[0], lat[0], color='#008000', marker='^', label='Start', zorder=2)
	plt.scatter(long[-1], lat[-1], color='red', marker='^', label='End', zorder=2)
	plt.legend()
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.title(f'Trajectory of test cyclone {cyclone_id}')
	plt.show()

def plot_cyclone_history_and_future(*, track_history: np.ndarray, track_future: np.ndarray, track_forecast: np.ndarray = None,
	save_to_file: bool = False, file_id: int = None, plot_over_world_map: bool = False) -> None:

	"""
	Plot history, true future, and forecast (if any) of a cyclone track.
	This function is used to plot training examples, or model output vs true trajectory.
	"""

	if plot_over_world_map and track_forecast is not None:
		min_lat = min([min(track_history[:, 0]), min(track_future[:, 0]), min(track_forecast[:, 0])])
		max_lat = max([max(track_history[:, 0]), max(track_future[:, 0]), max(track_forecast[:, 0])])
		min_long = min([min(track_history[:, 1]), min(track_future[:, 1]), min(track_forecast[:, 1])])
		max_long = max([max(track_history[:, 1]), max(track_future[:, 1]), max(track_forecast[:, 1])])

		plot_world_map(min_lat=min_lat - 2, max_lat=max_lat + 2, min_long=min_long - 2, max_long=max_long + 2)

	lat_history, long_history = track_history.T
	lat_future, long_future = track_future.T

	plt.plot(long_history, lat_history, color='#0080ff', label='Track history')
	plt.plot(long_future, lat_future, color='#0080ff', ls='--', label='True track future')

	if track_forecast is None:
		plt.title('History (x) and future (y) of a cyclone\n(training sample)')
	else:
		lat_forecast, long_forecast = track_forecast.T
		plt.plot(long_forecast, lat_forecast,
			color='red', ls=':', linewidth=2.5,
			label='Track forecast')

		lat_mae = mae(lat_forecast, lat_future)
		long_mae = mae(long_forecast, long_future)
		plt.title(f'History (x), true future (y), and model_v{VERSION_NUM} forecast of a cyclone'
			f'\nLatitude MAE: {lat_mae:.3f} deg'
			f'\nLongitude MAE: {long_mae:.3f} deg')

	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.legend()

	if save_to_file:
		plt.savefig(f'forecastoutputs/model_v{VERSION_NUM}_forecast_{file_id}.png')

	plt.show()

def plot_loss(history: History) -> None:
	"""Plot and save a training/validation loss graph"""

	final_train_loss = history.history['loss'][-1]
	final_val_loss = history.history['val_loss'][-1]
	plt.plot(history.history['loss'], label='Training')
	plt.plot(history.history['val_loss'], label='Validation')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('MSE loss')
	plt.title(f'model_v{VERSION_NUM} MSE loss during training'
		f'\nFinal training loss: {final_train_loss:.3f}'
		f'\nFinal validation loss: {final_val_loss:.3f}')
	plt.savefig(f'lossgraphs/model_v{VERSION_NUM}_loss.png')
	plt.show()
