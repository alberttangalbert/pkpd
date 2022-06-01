import tensorflow
import pk_data_utils as pkdu 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU

nsubj = 100
ntime_steps = 337
n_features = 3
file_name = "intensiveQD.csv"
dose2use = 0.03

pkpd_data = pkdu.read_in_data(file_name)[0]["Combo6: [0.1, 1.0, 0.1, 10.0]"]
pkpd_data = pkdu.normalize_doses(pkpd_data, [dose2use])
pkpd_data = pkpd_data["DOSE: " + str(dose2use)]

dosing_regimen = pkpd_data[:, :, 1].reshape(nsubj, ntime_steps, 1)
pk_data = pkpd_data[:, :, 3].reshape(nsubj, ntime_steps, 1)
pd_data = pkpd_data[:, :, 4].reshape(nsubj, ntime_steps, 1)
#%%
def custom_loss_function(y_true, y_pred):
   print(y_true)
   print()
   print(y_pred)
#%%
model = Sequential()
model.add(LSTM(ntime_steps, activation='relu', input_shape=(ntime_steps, n_features), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(RepeatVector(ntime_steps))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='custom_loss_function')
model.summary()