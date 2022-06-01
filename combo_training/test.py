import pandas as pd
import numpy as np
# import scipy as py
import os
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle
import math
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import L1L2
# import tensorflow and keras functions, some are useless in this running
import tensorflow
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
# from keras.layers import activations
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import backend as K
import time
#%%
#get current file path then create folder to output results
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
save_path = os.path.join(dir_path, "pkpd_combo")

np.random.seed(7)

#data specifications 
num_combos = 16
doses = [0.03, 0.1, 0.3, 1.0, 3.0 , 10.0, 30.0]
num_subjects = 100
timesteps = 43
features_to_train = 3
feature_to_predict = 3
combo_to_test = 3
doses_to_train = [0.03, 0.1]
tvratio = 9/10

#%%
cellunits = 43
model_stateless = Sequential()
model_stateless.add(Bidirectional(GRU(units=cellunits, return_sequences=True, kernel_initializer="orthogonal"),
                        input_shape=(None, features_to_train)))
#model_stateless.add(GRU(units=cellunits, return_sequences=True, 
#                        input_shape=[None, predict_features]))
# =============================================================================
# model_stateless.add(TimeDistributed(Dense(units=cellunits, activation="linear")))
# model_stateless.add(GRU(units=cellunits, return_sequences=True))
# =============================================================================
# =============================================================================
#model_stateless.add(TimeDistributed(Dense(units=cellunits, activation="linear")))
#model_stateless.add(GRU(units=cellunits, return_sequences=True))
# =============================================================================
model_stateless.add(TimeDistributed(Dense(1, activation = "linear")))
model_stateless.add(GRU(units=cellunits, return_sequences=True, kernel_initializer="orthogonal"))
model_stateless.add(TimeDistributed(Dense(1, activation = "linear")))
#model_stateless.add(TimeDistributed(Dense(units=cellunits, activation="linear")))
#model_stateless.add(Dense(1, activation="linear"))
#model_stateless.add(GRU(units=cellunits, return_sequences=True, 
                        #input_shape=[None, predict_features]))
#model_stateless.add(Dense(units=cellunits, activation="linear"))
model_stateless.add(TimeDistributed(Dense(1, activation="sigmoid")))
#model_stateless.add(Dense(1, activation="linear"))

model_stateless.compile(loss = 'mse', optimizer = 'adam')

bestmodel_filepath = r"C:\Users\alber\Code\Python\PKPD\pkpd_combo\combo6\bestmodel_epochs1000_train003_01_10.h5"
model_stateless.load_weights(bestmodel_filepath)
# =============================================================================
# #%%
# predict = model_stateless.predict(dataXY["Combo6: [0.1, 1.0, 0.1, 10.0]"]["DOSE: 3.0"][0])
# #
# title = "Combo6 - 0.03, 0.1, 0.3 to predict 1 GRU, 1 Dense"
# plt.title(title)
# X = dataXY["Combo6: [0.1, 1.0, 0.1, 10.0]"]["DOSE: 3.0"][0][0,:,2]
# Y = predict[0]
# plt.plot(X, Y)
# XT = dataXY["Combo6: [0.1, 1.0, 0.1, 10.0]"]["DOSE: 3.0"][0][0,:,2]
# YT = dataXY["Combo6: [0.1, 1.0, 0.1, 10.0]"]["DOSE: 3.0"][1][0]
# plt.plot(XT, YT)
# # =============================================================================
# # plt.plot(combos["Combo6: [0.1, 1.0, 0.1, 10.0]"]["DOSE: 0.03"][0,:,3])
# # true = dataXY["Combo6: [0.1, 1.0, 0.1, 10.0]"]["DOSE: 1.0"][1]
# # plt.plot(true[0])
# # plt.legend(['predict',  "0.03", 'actual'], loc='upper left')
# # =============================================================================
# plt.legend(['predict', 'actual'], loc='upper left')
# =============================================================================
predict = model_stateless.predict(dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.03"][0])
title = "Combo3 - 0.03, 0.1 to predict 0.3"
plt.title(title)
X = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.03"][0][0,:,2]
Y = predict[0]
plt.plot(Y)
#XT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.3"][0][0,:,2]
#YT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.3"][1][0]
#plt.plot(YT)


plt.legend(['predict', 'actual'], loc='upper left')
#%%
XT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.03"][0][0,:,2]
YT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.03"][1][0]
plt.plot(XT)
plt.plot(YT)
XT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.1"][0][0,:,2]
YT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.1"][1][0]
plt.plot(XT)
plt.plot(YT)
XT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][0][0,:,2]
YT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][1][0]
plt.plot(XT)
plt.plot(YT)
# =============================================================================
# YT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.3"][1][0]
# plt.plot(YT)
# YT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][1][0]
# plt.plot(YT)
# =============================================================================
#XT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.3"][0][0,:,2]
#YT = dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 0.3"][1][0]
#%%
# =============================================================================
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model accuracy')
# plt.xlabel('epoch')
# plt.legend(['accuracy', 'loss', "val_loss"], loc='upper left')
# plt.show()
# =============================================================================
#%%
plt.scatter(dataX0[0,:,2], dataY0[0,:,0])
for j in range(10):
    for i in range(len(dataX0[j])):
        plt.annotate(dataX0[j,i,1], (dataX0[j,i,2], dataY0[j,i,0]))





