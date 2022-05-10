"""
MSc Research Project: Cyclone Trajectory Forecasting

Data augmentation functionality

Author: Sam Barba
"""

from datetime import datetime
import numpy as np
from pandas import concat, DataFrame, date_range

from utils.constants import PANDAS_DEFAULT_DATETIME_FORMAT

def augment(df: DataFrame) -> DataFrame:
	"""
	Augment a DataFrame of a single cyclone with more feature values,
	such that there is a measurement every hour
	"""

	def change_datetime_format_to_pandas(s: str) -> str:
		"""
		Change datetime format of a string `s` from
		dd.mm.yyyy HH:MM to yyyy-mm-dd HH:MM:SS (pandas format)
		"""

		s_split = s.split(' ')
		day, month, year = s_split[0].split('.')
		hour, minute = s_split[1].split(':')

		return f'{year}-{month}-{day} {hour}:{minute}:00'

	def change_datetime_format_to_dmyhm(s: str) -> str:
		"""
		Change datetime format of a string `s` from
		yyyy-mm-dd HH:MM:SS (pandas) to dd.mm.yyyy HH:MM
		"""

		s_split = s.split(' ')
		year, month, day = s_split[0].split('-')
		hour, minute, _ = s_split[1].split(':')

		return f'{day}.{month}.{year} {hour}:{minute}'

	def insert_row_at_idx(df: DataFrame, idx: int, row_values: list) -> DataFrame:
		"""Insert a new row in a DataFrame at a specified index"""

		df1, df2 = df[:idx].copy(), df[idx:].copy()
		df1.loc[idx] = row_values

		df = concat([df1, df2])
		df.index = range(df.shape[0])
		return df

	"""
	1. Remove any non-hourly recordings (e.g. 11:45) for temporal consistency
	   (model must predict HOURLY cyclone movement)
	"""

	df.index = range(df.shape[0])
	rows_to_drop = []
	for row_idx in range(df.shape[0]):
		datetime_str = df.iloc[row_idx]['Datetime']
		hours_mins = datetime_str.split(' ')[1]
		# If it isn't an exact hour (i.e. minutes != '00')...
		if hours_mins[-2:] != '00':
			rows_to_drop.append(row_idx)

	if rows_to_drop:
		df = df.drop(rows_to_drop)
		df.index = range(df.shape[0])

	assert df.shape[0] >= 2, 'DataFrame has less than 2 rows, or rows have non-hourly recordings (e.g. 11:29am)'

	# Apply default pandas datetime format to avoid datetime augmentation problems
	df['Datetime'] = df['Datetime'].apply(change_datetime_format_to_pandas)

	"""
	2. Calculate time gaps between consecutive rows (in hours)
	   e.g. AL122015 (KATE),12.11.2015 12:00,30,900,41.3,-50.4
	   and  AL122015 (KATE),12.11.2015 14:00,30,900,41.4,-50.6
	   = 2 hrs
	"""

	time_gaps = []
	for i in range(df.shape[0] - 1):
		current_datetime, next_datetime = df.iloc[i:i + 2]['Datetime']

		# Convert datetime strings to datetime objects
		current_datetime = datetime.strptime(current_datetime, PANDAS_DEFAULT_DATETIME_FORMAT)
		next_datetime = datetime.strptime(next_datetime, PANDAS_DEFAULT_DATETIME_FORMAT)

		# Get interval in hours
		interval = next_datetime - current_datetime
		time_gaps.append(interval.total_seconds() / 3600)

	"""
	3. For each pair of existing rows, add N - 1 more rows in between, where N is the time gap (in hours) between rows.
	   Data to augment: datetime, maximum wind, minimum pressure, latitude, longitude.
	"""

	insert_idx = 0
	cyclone_id = df['ID (name)'].values[0]

	for time_gap in time_gaps:
		# rows_to_add = N - 1 (if N = 1, time difference is 1 hour, so no need to augment)
		rows_to_add = int(time_gap) - 1

		if rows_to_add < 1:
			insert_idx += 1
			continue

		current_datetime, next_datetime = df.iloc[insert_idx:insert_idx + 2]['Datetime']
		current_max_wind, next_max_wind = df.iloc[insert_idx:insert_idx + 2]['Maximum Wind']
		current_min_pressure, next_min_pressure = df.iloc[insert_idx:insert_idx + 2]['Minimum Pressure']
		current_lat, next_lat = df.iloc[insert_idx:insert_idx + 2]['Latitude']
		current_long, next_long = df.iloc[insert_idx:insert_idx + 2]['Longitude']

		# + 2 because pd.date_range and np.linspace include the start and end values,
		# which we don't need to add as they're already in the DF
		datetime_linspace = date_range(current_datetime, next_datetime, periods=rows_to_add + 2)
		wind_linspace = np.round(np.linspace(current_max_wind, next_max_wind, rows_to_add + 2), 3)
		pressure_linspace = np.round(np.linspace(current_min_pressure, next_min_pressure, rows_to_add + 2), 3)
		lat_linspace = np.round(np.linspace(current_lat, next_lat, rows_to_add + 2), 3)
		long_linspace = np.round(np.linspace(current_long, next_long, rows_to_add + 2), 3)

		# Exclude start and end values which are already in DF
		datetimes_to_add = [str(d) for d in datetime_linspace[1:-1]]
		winds_to_add = wind_linspace[1:-1]
		pressures_to_add = pressure_linspace[1:-1]
		lats_to_add = lat_linspace[1:-1]
		longs_to_add = long_linspace[1:-1]

		# Zip these together to form a list of rows to add
		new_rows = list(zip(datetimes_to_add, winds_to_add, pressures_to_add, lats_to_add, longs_to_add))
		# Add cyclone ID/name at start of each row
		new_rows = [[cyclone_id] + list(row) for row in new_rows]

		# Add the new rows (+ 1 because we want new rows AFTER the existing
		# row, instead of replacing it)
		for i, row in enumerate(new_rows):
			df = insert_row_at_idx(df, insert_idx + i + 1, row)

		# The next index at which to add data, given the dataframe has grown in length
		insert_idx += rows_to_add + 1

	"""
	4. Apply custom datetime format to datetime column
	"""

	df['Datetime'] = df['Datetime'].apply(change_datetime_format_to_dmyhm)

	return df
