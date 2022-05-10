"""
MSc Research Project: Cyclone Trajectory Forecasting

Model training functionality

Author: Sam Barba
"""

from keras.models import Sequential
from numpy import squeeze
from tensorflow.keras.callbacks import EarlyStopping

from data.train_data_prep import get_train_val_data
from graph_plotting import *
from utils.constants import VERSION_NUM
from utils.model_params import PARAMS

def train_model(model: Sequential, cyclone_dfs: list) -> None:
	"""Train a model on a set of cyclone dataframes"""

	# Load parameters of this model version

	history_size = PARAMS[VERSION_NUM]['history_size']
	target_size = PARAMS[VERSION_NUM]['target_size']
	min_delta = PARAMS[VERSION_NUM]['min_delta']
	patience = PARAMS[VERSION_NUM]['patience']
	epochs = PARAMS[VERSION_NUM]['epochs']

	# Convert dataframes to training and validation data,
	# given history_size of input and target_size to predict

	train_data, val_data = get_train_val_data(cyclone_dfs, history_size, target_size)

	# Plot an example section of a cyclone training trajectory

	for x, y in train_data.take(1):
		# Plot the first ([0]) history_size hours (x) and target_size hours (y)
		# of lat/long values (i.e. only last 2 columns - don't need wind speed or pressure)
		plot_cyclone_history_and_future(track_history=squeeze(x)[0][:, -2:],
			track_future=squeeze(y)[0][:, -2:])

	# Define early stopping:
	# - min_delta = min. change in monitored quality to qualify as an improvement
	# - patience = no. epochs with no improvement after which training will stop
	# - restore_best_weights = whether to restore model weights from the epoch with
	# 	the best value of the monitored quantity (validation loss in this case)

	early_stopping = EarlyStopping(monitor='val_loss',
		min_delta=min_delta,
		patience=patience,
		restore_best_weights=True)

	# Train model

	history = model.fit(train_data,
		validation_data=val_data,
		epochs=epochs,
		callbacks=[early_stopping],
		verbose=1)

	# Plot and save loss graph for this model version (in /lossgraphs)

	plot_loss(history)
