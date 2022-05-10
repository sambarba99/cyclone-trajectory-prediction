"""
MSc Research Project: Cyclone Trajectory Forecasting

Calculator for heading between 2 coordinates (source: https://www.movable-type.co.uk/scripts/latlong.html)

Author: Sam Barba
"""

from numpy import radians, arctan2, sin, cos, degrees

def calc_heading(*, lat1: float, long1: float, lat2: float, long2: float) -> float:
	"""Calculate the heading (deg) between 2 coordinates"""

	# Convert coords to radians
	lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])

	d = long2 - long1
	cos_lat2 = cos(lat2)
	heading = arctan2(sin(d) * cos_lat2, cos(lat1) * sin(lat2) - sin(lat1) * cos_lat2 * cos(d))

	return degrees(heading)
