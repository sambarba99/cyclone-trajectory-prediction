"""
MSc Research Project: Cyclone Trajectory Forecasting

Parameters per model version

Author: Sam Barba
"""

PARAMS = {1:                 # Parameters of model version 1
	{'history_size': 48,     # Using the past 48 hours of data...
	'target_size': 12,       # ...learn to predict the next 12 hours
	'learning_rate': 1e-3,
	'epochs': 50,            # Number of training iterations
	'min_delta': 0,          # See model_training.py for explanation of min_delta and patience
	'patience': 10},
	2:                       # Parameters of model version 2, etc.
	{'history_size': 48,
	'target_size': 12,
	'learning_rate': 5e-3,
	'epochs': 50,
	'min_delta': 0,
	'patience': 10},
	3:
	{'history_size': 48,
	'target_size': 12,
	'learning_rate': 5e-4,
	'epochs': 150,
	'min_delta': 0,
	'patience': 10},
	4:
	{'history_size': 48,  # 96
	'target_size': 12,
	'learning_rate': 1e-4,
	'epochs': 150,
	'min_delta': 0,
	'patience': 10}
}
