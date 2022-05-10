"""
MSc Research Project: Cyclone Trajectory Forecasting

Calculator for Great Circle distance between 2 coordinates (source: https://www.movable-type.co.uk/scripts/latlong.html)

Author: Sam Barba
"""

from numpy import radians, arccos, sin, cos

from utils.constants import EARTH_RADIUS

def great_circle_distance(*, lat1: float, long1: float, lat2: float, long2: float) -> float:
	"""
	Calculate the 'Great Circle' distance (km) along Earth's surface
	between 2 coordinates
	"""

	if abs(lat1 - lat2) == 0 and abs(long1 - long2) == 0:
		return 0

	# Convert coords to radians
	lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])

	return EARTH_RADIUS * (arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(long1 - long2)))
