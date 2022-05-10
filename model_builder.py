"""
MSc Research Project: Cyclone Trajectory Forecasting

Model builder functionality (see /modelplots for visualisation).
The match-case statement highlights major changes between consecutive versions,
corresponding to section 4 in the report.

Author: Sam Barba
"""

from keras.layers import BatchNormalization, Bidirectional, ConvLSTM1D, ConvLSTM2D, Dense, Dropout, \
	Flatten, Input, LeakyReLU, LSTM, RepeatVector, Reshape, TimeDistributed
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from utils.constants import VERSION_NUM
from utils.model_params import PARAMS

def build_model(*, input_timesteps: int, output_timesteps: int, num_features: int = 4) -> Sequential:
	"""
	Construct the model for cyclone trajectory forecasting.
	`num_features` is 4 by definition: maximum wind, minimum pressure, latitude, and longitude.
	The latter 2 are the features of interest to predict; the others are supplementary,
	as explained in section 3.1.
	"""

	model = Sequential(name=f'model_v{VERSION_NUM}')
	input_shape = None

	match VERSION_NUM:
		case 1:
			input_shape = (input_timesteps, num_features, 1, 1)

			# 'Encoder' model (see 4.2.1)
			model.add(BatchNormalization(input_shape=input_shape))
			model.add(ConvLSTM2D(filters=64, kernel_size=(10, 1), padding='same', activation='elu'))
			model.add(Dropout(0.7))
			model.add(Flatten())
			model.add(RepeatVector(output_timesteps))
			model.add(Reshape((output_timesteps, num_features, 1, 64)))

			# 'Decoder' model
			model.add(ConvLSTM2D(filters=128, kernel_size=(10, 1), padding='same', activation='elu', return_sequences=True))
			model.add(Dropout(0.8))
			model.add(TimeDistributed(Dense(units=num_features, activation='relu', name='output')))

		case 2:
			input_shape = (input_timesteps, num_features, 1)

			model.add(BatchNormalization(input_shape=input_shape))
			model.add(ConvLSTM1D(filters=64, kernel_size=10, padding='same', activation='elu'))
			model.add(Flatten())
			model.add(RepeatVector(output_timesteps))

			model.add(LSTM(units=8, activation='elu', return_sequences=True))
			model.add(TimeDistributed(Dense(units=num_features, name='output')))  # Activation = linear (default)

		case 3:
			input_shape = (input_timesteps, num_features)

			model.add(LSTM(units=128, activation='elu', input_shape=input_shape))
			model.add(Dense(units=128, activation='elu'))
			model.add(RepeatVector(output_timesteps))

			model.add(LSTM(units=8, activation='elu', return_sequences=True))
			model.add(Dense(units=8, activation='elu'))
			model.add(TimeDistributed(Dense(units=num_features, name='output')))

		case 4:
			input_shape = (input_timesteps, num_features)

			model.add(Input(shape=input_shape))
			model.add(Bidirectional(LSTM(units=256)))
			# Advanced activations in Keras must be called as layers like this,
			# rather than being passed as aliases
			model.add(LeakyReLU())
			model.add(RepeatVector(output_timesteps))

			model.add(Bidirectional(LSTM(units=128)))
			model.add(LeakyReLU())
			model.add(RepeatVector(output_timesteps))

			model.add(Bidirectional(LSTM(units=128)))
			model.add(LeakyReLU())
			model.add(RepeatVector(output_timesteps))

			model.add(Bidirectional(LSTM(units=128)))
			model.add(LeakyReLU())
			model.add(RepeatVector(output_timesteps))

			model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
			model.add(LeakyReLU())
			model.add(TimeDistributed(Dense(units=num_features, name='output')))

	# See section 4.1
	optim = Adam(learning_rate=PARAMS[VERSION_NUM]['learning_rate'])
	model.compile(loss='mse', optimizer=optim)
	model.build(input_shape=input_shape)

	return model
