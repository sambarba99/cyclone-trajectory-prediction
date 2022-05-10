"""
MSc Research Project: Cyclone Trajectory Forecasting

Functionality for preparing training/validation data

Author: Sam Barba
"""

from numpy import array
from pandas import DataFrame
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import ShuffleDataset, TensorSliceDataset

from utils.constants import FEATURES_TO_PREDICT, TRAIN_PROP, VERSION_NUM

def convert_data_to_history_and_target(dataset: DataFrame, history_size: int, target_size: int) -> tuple[list, list]:
	"""
	Convert a dataset to x (historical data) and y (future data to predict),
	given a history size and size of forecast target ('sliding window' method, report section 2.1)
	"""

	assert len(dataset) >= history_size + target_size, \
		'Dataset length must be >= history_size + target_size' + \
		f'\n(got dataset length = {len(dataset)}, history_size + target_size = {history_size + target_size})'

	start_idx = history_size
	end_idx = len(dataset) - target_size + 1

	x = [dataset[i - history_size:i] for i in range(start_idx, end_idx)]
	y = [dataset[i:i + target_size] for i in range(start_idx, end_idx)]

	return x, y

def get_train_val_data(cyclone_dfs: list, history_size: int, target_size: int) -> tuple[ShuffleDataset, TensorSliceDataset]:
	"""Convert a collection of cyclone dataframes to sets of training/validation"""

	# Trim cyclone dataframes that are too long, to make them all the same length
	# (training/validation arrays cannot be 'jagged')
	min_length = min([df.shape[0] for df in cyclone_dfs])
	trimmed_dfs = [df[:min_length] for df in cyclone_dfs]

	# Keep only interested features
	trimmed_dfs = [df[FEATURES_TO_PREDICT].values for df in trimmed_dfs]

	# Split into training and validation
	split = int(len(trimmed_dfs) * TRAIN_PROP)
	train_dfs, val_dfs = trimmed_dfs[:split], trimmed_dfs[split:]

	# Convert these to history (x) and forecast target (y),
	# and put them into respective training/validation arrays
	x_train, y_train = [], []
	for df in train_dfs:
		xt, yt = convert_data_to_history_and_target(df, history_size, target_size)
		x_train.append(xt)
		y_train.append(yt)
	x_val, y_val = [], []
	for df in val_dfs:
		xv, yv = convert_data_to_history_and_target(df, history_size, target_size)
		x_val.append(xv)
		y_val.append(yv)

	# Convert to np arrays
	x_train, y_train, x_val, y_val = map(array, [x_train, y_train, x_val, y_val])

	# Model versions 1 and 2 need data shaped as follows before training
	match VERSION_NUM:
		case 1:
			x_train = x_train.reshape((*x_train.shape, 1, 1))
			y_train = y_train.reshape((*y_train.shape, 1, 1))
			x_val = x_val.reshape((*x_val.shape, 1, 1))
			y_val = y_val.reshape((*y_val.shape, 1, 1))
		case 2:
			x_train = x_train.reshape((*x_train.shape, 1))
			y_train = y_train.reshape((*y_train.shape, 1))
			x_val = x_val.reshape((*x_val.shape, 1))
			y_val = y_val.reshape((*y_val.shape, 1))

	# Cache and shuffle training data
	buffer_size = len(train_dfs) * 100
	train_data = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
		.cache()                # Cache dataset in memory (avoids repetitive preprocessing transformations)
		.shuffle(buffer_size))  # Shuffle samples so they're always random when fed to the model
	# (buffer_size = num. elements from original dataset from which the new dataset will sample:
	# having this = 1 means no shuffling; having it > len(train_data) means uniform shuffling)

	val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))

	return train_data, val_data
