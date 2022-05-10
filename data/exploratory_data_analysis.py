"""
MSc Research Project: Cyclone Trajectory Forecasting

Cyclone exploratory data analysis functions

Author: Sam Barba
"""

from datetime import datetime
from numpy import mean, std
from pandas import DataFrame

from graph_plotting import plot_attribute_bar_graph_multi_cyclones
from utils.constants import CUSTOM_DATETIME_FORMAT
from utils.distance_calc import great_circle_distance
from utils.heading_calc import calc_heading

def get_cyclone_duration(cyclone_df: DataFrame) -> tuple[str, float]:
	"""Get the duration of a cyclone (hours) using the first and last measurements"""

	cyclone_id = cyclone_df['ID (name)'].values[0]

	if cyclone_df.shape[0] == 1: return cyclone_id, 0

	measurement_start = cyclone_df.iloc[0]['Datetime']
	measurement_end = cyclone_df.iloc[-1]['Datetime']
	start_datetime = datetime.strptime(measurement_start, CUSTOM_DATETIME_FORMAT)
	end_datetime = datetime.strptime(measurement_end, CUSTOM_DATETIME_FORMAT)

	# Get interval in hours, and return it with cyclone ID (name)
	interval = end_datetime - start_datetime
	duration_hours = interval.total_seconds() / 3600
	return cyclone_id, duration_hours

def do_eda(cyclone_dfs: list, *, all_cyclones: bool) -> None:
	"""Do varaious exploratory data analyses on a list of cyclone dataframes"""

	def get_cyclone_initial_heading(cyclone_df: DataFrame) -> float:
		"""Get the initial heading of a cyclone using the first 2 measurements"""

		lat1, lat2 = cyclone_df[['Latitude']].values[:2]
		long1, long2 = cyclone_df[['Longitude']].values[:2]
		return calc_heading(lat1=lat1, long1=long1, lat2=lat2, long2=long2)

	def get_cyclone_displacement(cyclone_df: DataFrame) -> tuple[str, float]:
		"""Calculate displacement of a cyclone (km) using the first and last measurements"""

		cyclone_track = cyclone_df[['Latitude', 'Longitude']].values

		if len(cyclone_track) < 2:
			disp = 0
		else:
			start_lat, start_long = cyclone_track[0]
			end_lat, end_long = cyclone_track[-1]
			disp = great_circle_distance(lat1=start_lat, long1=start_long, lat2=end_lat, long2=end_long)

		return cyclone_df['ID (name)'].values[0], disp

	def get_cyclone_mean_speed(cyclone_df: DataFrame) -> tuple[str, float]:
		"""Get the mean speed (km/h) of a cyclone"""

		cyclone_id = cyclone_df['ID (name)'].values[0]

		if cyclone_df.shape[0] == 1: return cyclone_id, 0

		# Calculate time gaps and geographic displacements
		# between consecutive rows (in hours and km, respectively).
		# E.g. AL122015 (KATE),12.11.2015 12:00,41.3,-50.4
		# and AL122015 (KATE),12.11.2015 14:00,41.4,-50.6
		# = 2 hrs, 20.1km

		time_gaps, displacement_gaps = [], []
		for i in range(cyclone_df.shape[0] - 1):
			current_datetime, next_datetime = cyclone_df.iloc[i:i + 2]['Datetime']

			# Convert datetime strings to datetime objects
			current_datetime = datetime.strptime(current_datetime, CUSTOM_DATETIME_FORMAT)
			next_datetime = datetime.strptime(next_datetime, CUSTOM_DATETIME_FORMAT)

			# Get interval in hours
			interval = next_datetime - current_datetime
			time_gaps.append(interval.total_seconds() / 3600)

			# Get displacement in km
			current_lat, next_lat = cyclone_df.iloc[i:i + 2]['Latitude']
			current_long, next_long = cyclone_df.iloc[i:i + 2]['Longitude']
			disp = great_circle_distance(lat1=current_lat, long1=current_long,
				lat2=next_lat, long2=next_long)
			displacement_gaps.append(disp)

		# Get list of speeds using these lists
		speeds = [d / t for d, t in zip(displacement_gaps, time_gaps)]

		# Return the mean of these speeds and the cyclone ID (name)
		return cyclone_id, mean(speeds)

	# Print cyclone genesis info

	print('----- CYCLONE GENESIS INFO -----\n')
	genesis_lats = [df['Latitude'].values[0] for df in cyclone_dfs]
	genesis_longs = [df['Longitude'].values[0] for df in cyclone_dfs]
	print('Mean genesis location of a cyclone: '
		f'{mean(genesis_lats):.1f}, {mean(genesis_longs):.1f}')
	print('Genesis latitude standard deviation: '
		f'{std(genesis_lats):.1f} deg')
	print('Genesis longitude standard deviation: '
		f'{std(genesis_longs):.1f} deg')

	# Print cyclone NSEW extremes info

	print('\n----- CYCLONE NSEW EXTREMES INFO -----\n')
	most_north = max([max(df['Latitude'].values) for df in cyclone_dfs])
	most_south = min([min(df['Latitude'].values) for df in cyclone_dfs])
	most_east = max([max(df['Longitude'].values) for df in cyclone_dfs])
	most_west = min([min(df['Longitude'].values) for df in cyclone_dfs])
	print(f'Northernmost position of a cyclone: {most_north} deg')
	print(f'Southernmost position of a cyclone: {most_south} deg')
	print(f'Easternmost position of a cyclone: {most_east} deg')
	print(f'Westernmost position of a cyclone: {most_west} deg')

	# Print cyclone initial heading info

	cyclone_initial_headings = [get_cyclone_initial_heading(df)
		for df in cyclone_dfs if df.shape[0] >= 2]
	print('\n----- CYCLONE INITIAL HEADING INFO -----\n')
	print('Mean initial heading of a cyclone: '
		f'{mean(cyclone_initial_headings):.1f} deg')
	print('Initial heading standard deviation: '
		f'{std(cyclone_initial_headings):.1f} deg')

	# Print cyclone duration info

	cyclone_durations = [get_cyclone_duration(df) for df in cyclone_dfs]
	cyclone_durations = {cyclone_id: dur for cyclone_id, dur
		in cyclone_durations}
	cyclone_durations_nonzero = {cyclone_id: dur for cyclone_id, dur
		in cyclone_durations.items() if dur > 0}
	print('\n----- CYCLONE DURATION INFO -----\n')
	print('Shortest non-zero duration of a cyclone: '
		f'{round(min(cyclone_durations_nonzero.values()))} hrs '
		f'({min(cyclone_durations_nonzero, key=cyclone_durations_nonzero.get)})')
	print(f'Longest duration of a cyclone: {round(max(cyclone_durations.values()))} hrs '
		f'({max(cyclone_durations, key=cyclone_durations.get)})')
	print('Mean duration of a cyclone: '
		f'{round(mean(list(cyclone_durations.values())))} hrs')
	print('Duration standard deviation: '
		f'{std(list(cyclone_durations.values())):.1f} hrs')

	plot_attribute_bar_graph_multi_cyclones(
		cyclone_attributes=cyclone_durations_nonzero.values(),
		measured_quantity='duration',
		unit='hours',
		for_all_cyclones=all_cyclones
	)

	# Print cyclone dislpacement info

	print('\n----- CYCLONE DISPLACEMENT INFO -----\n')
	cyclone_displacements = [get_cyclone_displacement(df) for df in cyclone_dfs]
	cyclone_displacements = {cyclone_id: disp for cyclone_id, disp
		in cyclone_displacements}
	cyclone_displacements_nonzero = {cyclone_id: disp for cyclone_id, disp
		in cyclone_displacements.items() if disp > 0}
	print('Smallest non-zero displacement of a cyclone: '
		f'{min(cyclone_displacements_nonzero.values()):.1f} km '
		f'({min(cyclone_displacements_nonzero, key=cyclone_displacements_nonzero.get)})')
	print(f'Largest displacement of a cyclone: {max(cyclone_displacements.values()):.1f} km '
		f'({max(cyclone_displacements, key=cyclone_displacements.get)})')
	print('Mean displacement of a cyclone: '
		f'{mean(list(cyclone_displacements.values())):.1f} km')
	print('Displacement standard deviation: '
		f'{std(list(cyclone_displacements.values())):.1f} km')

	plot_attribute_bar_graph_multi_cyclones(
		cyclone_attributes=cyclone_displacements_nonzero.values(),
		measured_quantity='displacement',
		unit='km',
		for_all_cyclones=all_cyclones
	)

	# Print cyclone speed info

	print('\n----- CYCLONE SPEED INFO -----\n')
	cyclone_speeds = [get_cyclone_mean_speed(df) for df in cyclone_dfs]
	cyclone_speeds = {cyclone_id: speed for cyclone_id, speed
		in cyclone_speeds}
	cyclone_speeds_nonzero = {cyclone_id: speed for cyclone_id, speed
		in cyclone_speeds.items() if speed > 0}
	print('Smallest non-zero speed of a cyclone: '
		f'{min(cyclone_speeds_nonzero.values()):.1f} km/h '
		f'({min(cyclone_speeds_nonzero, key=cyclone_speeds_nonzero.get)})')
	print(f'Greatest speed of a cyclone: {max(cyclone_speeds.values()):.1f} km/h '
		f'({max(cyclone_speeds, key=cyclone_speeds.get)})')
	print('Mean speed of a cyclone: '
		f'{mean(list(cyclone_speeds.values())):.1f} km/h')
	print('Speed standard deviation: '
		f'{std(list(cyclone_speeds.values())):.1f} km/h')

	plot_attribute_bar_graph_multi_cyclones(
		cyclone_attributes=cyclone_speeds_nonzero.values(),
		measured_quantity='mean speed',
		unit='km/h',
		for_all_cyclones=all_cyclones
	)

	# Print cyclone maximum wind info

	print('\n----- CYCLONE MAXIMUM WIND INFO -----\n')
	# Only consider dataframes with complete values (-999 is a placeholder for non-existent values)
	cyclone_mean_max_winds = [(df['ID (name)'].values[0], mean(df['Maximum Wind'].values))
		for df in cyclone_dfs]
	cyclone_mean_max_winds = {cyclone_id: mean_max_wind for cyclone_id, mean_max_wind
		in cyclone_mean_max_winds}
	print('Smallest mean maximum wind of a cyclone: '
		f'{min(cyclone_mean_max_winds.values()):.1f} knots '
		f'({min(cyclone_mean_max_winds, key=cyclone_mean_max_winds.get)})')
	print('Greatest mean maximum wind of a cyclone: '
		f'{max(cyclone_mean_max_winds.values()):.1f} knots '
		f'({max(cyclone_speeds, key=cyclone_speeds.get)})')
	print('Mean maximum wind of a cyclone: '
		f'{mean(list(cyclone_mean_max_winds.values())):.1f} knots')
	print('Standard deviation of mean maximum winds: '
		f'{std(list(cyclone_mean_max_winds.values())):.1f} knots')

	plot_attribute_bar_graph_multi_cyclones(
		cyclone_attributes=cyclone_mean_max_winds.values(),
		measured_quantity='maximum wind',
		unit='knots',
		for_all_cyclones=all_cyclones
	)

	# Print cyclone minimum pressure info

	print('\n----- CYCLONE MINIMUM PRESSURE INFO -----\n')
	cyclone_mean_min_pressures = [(df['ID (name)'].values[0], mean(df['Minimum Pressure'].values))
		for df in cyclone_dfs]
	cyclone_mean_min_pressures = {cyclone_id: mean_min_pressure for cyclone_id, mean_min_pressure
		in cyclone_mean_min_pressures}
	print('Smallest mean minimum pressure of a cyclone: '
		f'{min(cyclone_mean_min_pressures.values()):.1f} millibar '
		f'({min(cyclone_mean_min_pressures, key=cyclone_mean_min_pressures.get)})')
	print('Greatest mean minimum pressure of a cyclone: '
		f'{max(cyclone_mean_min_pressures.values()):.1f} millibar '
		f'({max(cyclone_mean_min_pressures, key=cyclone_mean_min_pressures.get)})')
	print('Mean minimum pressure of a cyclone: '
		f'{mean(list(cyclone_mean_min_pressures.values())):.1f} millibar')
	print('Standard deviation of mean minimum pressures: '
		f'{std(list(cyclone_mean_min_pressures.values())):.1f} millibar')

	plot_attribute_bar_graph_multi_cyclones(
		cyclone_attributes=cyclone_mean_min_pressures.values(),
		measured_quantity='minimum pressure',
		unit='millibar',
		for_all_cyclones=all_cyclones
	)
