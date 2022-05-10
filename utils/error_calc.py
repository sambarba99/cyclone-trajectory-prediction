"""
MSc Research Project: Cyclone Trajectory Forecasting

Error calculation functionality

Author: Sam Barba
"""

import numpy as np

def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
	"""Mean absolute error"""
	return np.abs(predicted - actual).sum() / len(actual)
