#useful python libraries 
import numpy as np
import os
import time
import pickle
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.regularizers import L1L2
import tensorflow
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dropout, RepeatVector
from keras.layers import Dense
from keras.layers import LSTM, GRU, Bidirectional, Attention
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import backend as K
#custom libraries 
from data_parse import read_in_data, create_data_XY, create_tv_XY, normalize_QD_BID, normalize_combos
from train_func import train_combo

#change current working directory to current file path
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
combo_to_use = 6
tvratio = 9/10
training_dosing_regimen = "QD"
testing_dosing_regimen = "BID"
#%%
dose_combos = [[0.03, 0.1, 10.0, 30.0]]
 #%%
#parse training and test data sets 
#combos is the entire data set
QDcombos, combo_to_key = read_in_data("sdtab100QD.csv", training_dosing_regimen)
BIDcombos, combo_to_key = read_in_data("sdtab1000BID.csv", testing_dosing_regimen)
QDcombos, BIDcombos = normalize_QD_BID(QDcombos, BIDcombos) 

#QDcombos = normalize_combos(QDcombos)
#create folder for storing the training combo weights 
fileName = "\\" + training_dosing_regimen + "combo" + str(combo_to_use)
path = save_path + fileName
if not os.path.exists(path):
    os.makedirs(path)

#combo is the sub data set that is the focus 
combo_to_train_data = QDcombos[combo_to_key[combo_to_use]]
combo_to_test_data = BIDcombos[combo_to_key[combo_to_use]]

training_dataX, training_dataY = create_data_XY(combo_to_train_data, doses, features_to_train, feature_to_predict)
#split BID data into X and Y 
testing_dataX, testing_dataY = create_data_XY(combo_to_test_data, doses, features_to_train, feature_to_predict)

maxPKIdx = [0 for i in range(len(doses))]
maxPK = [0 for i in range(len(doses))]
mgreaterIdx = [[] for i in range(len(doses))]
for i in range(len(doses)):
    dkey = "DOSE: " + str(doses[i])
    for j in range(num_subjects):
        maxVal = max(combo_to_train_data[dkey][j,:,2])
        if maxPK[i] < maxVal:
            maxPK[i] = maxVal
            maxPKIdx[i] = j
        if i > 0 and maxVal > maxPK[i - 1]:
            mgreaterIdx[i].append(j)
#%%
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for doses_to_use in dose_combos:
    #split data into training and validation 
    training_dataX, training_dataY = create_data_XY(combo_to_train_data, doses_to_use, features_to_train, feature_to_predict)
    trainX, trainY, validX, validY = create_tv_XY(training_dataX, training_dataY, tvratio)
    
    #split BID data into X and Y 
    testing_dataX, testing_dataY = create_data_XY(combo_to_test_data, doses_to_use, features_to_train, feature_to_predict)
    #set training parameters
    num_epochs = 10000
    batch_size = 30

    #create a file name acceptable string representation of the doses without
    string_of_training_doses = ""
    for dose in doses_to_use:
        string_of_training_doses += str(dose) + "_"
    string_of_training_doses = string_of_training_doses[:-1].replace(".", "")
    
    #create file to store best model
    bestmodel_filename = "epochs{0}_train{1}.h5".format(num_epochs, string_of_training_doses)
    bestmodel_filepath = os.path.join(path, bestmodel_filename)
    
    #start training 
    history = train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, 
                                         save_path, combo_to_use, bestmodel_filepath)

   #%%
#testing 
#t = "withfirst"
#t = "runalldosecombos"
cellunits = 42
doses_to_train = [0.03]
string_of_training_doses = ""
for dose in doses_to_train:
    string_of_training_doses += str(dose) + "_"
traind = string_of_training_doses[:-1].replace(".", "")

model_stateless = Sequential()
model_stateless.add(GRU(cellunits, return_sequences = True,
                        input_shape=(None, features_to_train)))
model_stateless.add(TimeDistributed(Dense(cellunits, activation = "ELU")))
model_stateless.add(GRU(cellunits, return_sequences=True))
model_stateless.add(TimeDistributed(Dense(cellunits, activation = "ELU")))
model_stateless.add(GRU(cellunits, return_sequences=True))
model_stateless.add(TimeDistributed(Dense(cellunits, activation = "ELU")))
model_stateless.add(TimeDistributed(Dense(1, activation = "sigmoid")))

model_stateless.compile(loss = 'mse', optimizer = 'adam')
#model_stateless.load_weights(bestmodel_filepath)
model_stateless.load_weights(r"pkpd_combo/QDcombo6/epochs15000_train{0}.h5".format(traind))
#model_stateless.load_weights(r"C:\Users\huado\Documents\Python\pkpd\gru\pkpd_combo\QDcombo6\epochs5000_train300.h5")
#%%
subject = 17
doses_to_test = [0.03,0.1]

plt.title("Train on {0} to predict {1} subject {2}".format(doses_to_train,doses_to_test,subject))

for dosett in doses_to_test:
    dose_key = "DOSE: " + str(dosett)

    predict = model_stateless.predict(combo_to_train_data[dose_key][:,:,:3])
    true = combo_to_train_data[dose_key][:,:,3]

    plt.plot(true[subject])
    plt.plot(predict[subject])
# =============================================================================
# plt.legend(['true 30.0', 'predict 30.0', 'true 10.0', 'predict 10.0', 'true 3.0', 
#             'predict 3.0', 'true 1.0', 'predict 1.0', 'true 0.3', 'predict 0.3',
#             'true 0.1', 'predict 0.1', 'true 0.03', 'predict 0.03'], loc='center left', bbox_to_anchor=(1, 0.7))
# =============================================================================
plt.legend(['true 0.03', 'predict 0.03', 'true 0.1', 'predict 0.1', 'true 0.3', 
            'predict 0.3', 'true 1.0', 'predict 1.0', 'true 3.0', 'predict 3.0',
            'true 10.0', 'predict 10.0', 'true 30.0', 'predict 30.0'], loc='center left', bbox_to_anchor=(1, 0.7))
# =============================================================================
# 
# dose_key = "DOSE: 0.1"
# 
# predict = model_stateless.predict(combo_to_train_data[dose_key][:,:,:3])
# true = combo_to_train_data[dose_key][:,:,3]
# 
# plt.plot(predict[subject])
# plt.plot(true[subject])
# 
# dose_key = "DOSE: 0.3"
# 
# predict = model_stateless.predict(combo_to_train_data[dose_key][:,:,:3])
# true = combo_to_train_data[dose_key][:,:,3]
# 
# plt.plot(predict[subject])
# plt.plot(true[subject])
# 
# dose_key = "DOSE: 1.0"
# 
# predict = model_stateless.predict(combo_to_train_data[dose_key][:,:,:3])
# true = combo_to_train_data[dose_key][:,:,3]
# 
# plt.plot(predict[subject])
# plt.plot(true[subject])
# 
# =============================================================================
























