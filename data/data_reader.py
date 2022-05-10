"""
MSc Research Project: Cyclone Trajectory Forecasting

Cyclone CSV data reading/cleaning functionality

Author: Sam Barba
"""

from pandas import concat, DataFrame, read_csv

from utils.constants import COLS_TO_KEEP, DATA_PATH

def separate_cyclone_dfs(df: DataFrame) -> list:
	"""
	Separate a DataFrame of multiple cyclones into sub-DataFrames,
	according to the ID (name) column
	"""

	cyclone_ids = sorted(list(set(df['ID (name)'].values)))

	# Cyclone dataframes
	cyclone_dfs = [df.loc[df['ID (name)'] == unique_id] for unique_id in cyclone_ids]

	return cyclone_dfs

def clean_raw_data() -> DataFrame:
	"""Read cyclone CSV data and clean it"""

	def format_datetime_string(s: str) -> str:
		"""
		Format a datetime string, e.g.:

		'201106251600' (25th June 2011, 16:00)
		->
		25.06.2011 16:00
		"""

		year, month, day = s[:4], s[4:6], s[6:8]
		hour, minute = s[8:10], s[10:]

		return f'{day}.{month}.{year} {hour}:{minute}'

	def format_lat_long_column(s: str) -> str:
		"""
		Format a coordinate string, e.g.:

		- '5.4E' -> 5.4
		- '2.3S' -> -2.3
		"""

		num_coord = float(s[:-1])
		return num_coord if s[-1] in ('N', 'E') else -num_coord

	"""
	1. Keep only necessary columns
	"""

	df = read_csv(DATA_PATH)
	df = df[COLS_TO_KEEP]

	"""
	2. Combine ID and name columns
	"""

	# E.g. ID = AL122015, name = KATE -> new column = AL122015 (KATE)
	df['ID (name)'] = df['ID'] + df['Name'].apply(lambda s: f' ({s})')

	# Drop old columns
	df = df.drop(['ID', 'Name'], axis=1)

	"""
	3. Combine date and time columns
	"""

	# Add leading zeroes to time column to pad out any short values
	df['Time'] = df['Time'].apply(lambda s: '{0:0>4}'.format(s))
	df['Datetime'] = df['Date'].astype(str) + df['Time'].astype(str)
	df['Datetime'] = df['Datetime'].apply(format_datetime_string)

	# Drop old columns
	df = df.drop(['Date', 'Time'], axis=1)

	"""
	4. Convert coordinate columns to float values by changing
	   N(E) and S(W) to +(-)
	"""

	df['Latitude'] = df['Latitude'].apply(format_lat_long_column)
	df['Longitude'] = df['Longitude'].apply(format_lat_long_column)

	# Reorder columns
	df = df[['ID (name)', 'Datetime', 'Maximum Wind', 'Minimum Pressure', 'Latitude', 'Longitude']]

	"""
	5. Separate cyclones into their own sub-dataframes according to the 'ID (name)' column,
	   and remove ones with missing wind speed or pressure values
	"""

	cyclone_dfs = separate_cyclone_dfs(df)

	# Remove dataframes with missing wind speed or pressure values
	# (-999 is a placeholder for missing values)
	cyclone_dfs = [df for df in cyclone_dfs
		if -999 not in df['Maximum Wind'].values
		and -999 not in df['Minimum Pressure'].values]

	"""
	6. Concatenate them back together now that dataframes with missing values are excluded
	"""

	df = concat(cyclone_dfs)
	df.index = range(df.shape[0])
	return df
