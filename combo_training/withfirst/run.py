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
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM, GRU, Attention
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
num_subjects = 10
timesteps = 43
features_to_train = 3
feature_to_predict = 3
combo_to_use = 6
tvratio = 9/10
training_dosing_regimen = "QD"
testing_dosing_regimen = "BID"
#%%
dose_combos = []
cdose_combo = []
for i in reversed(range(1,len(doses))):
    cdose_combo.append(doses[i])
    dose_combos.append([*cdose_combo])
dose_combos.append([0.03, 30.0])
dose_combos.append([0.03, 0.1, 10.0, 30.0])
dose_combos.append([0.03, 0.1, 30.0])
dose_combos.append([0.03, 10.0, 30.0])
dose_combos.append([0.1, 0.3, 1.0])
dose_combos.append([0.3, 1.0, 3.0])
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
#%%
for doses_to_use in dose_combos:
    #split data into training and validation 
    training_dataX, training_dataY = create_data_XY(combo_to_train_data, doses_to_use, features_to_train, feature_to_predict)
    trainX, trainY, validX, validY = create_tv_XY(training_dataX, training_dataY, tvratio)
    
    #split BID data in|to X and Y 
    testing_dataX, testing_dataY = create_data_XY(combo_to_test_data, doses_to_use, features_to_train, feature_to_predict)
    #set training parameters
    num_epochs = 5000
    batch_size = 10
    
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
cellunits = 42
doses_to_train = [30.0]
string_of_training_doses = ""
for dose in doses_to_train:
    string_of_training_doses += str(dose) + "_"
traind = string_of_training_doses[:-1].replace(".", "")

model_stateless = Sequential()
model_stateless.add(GRU(units=cellunits, return_sequences = True,
                        input_shape=(None, features_to_train)))
model_stateless.add(TimeDistributed(Dense(units=cellunits, activation = "ELU")))
model_stateless.add(GRU(units=cellunits, return_sequences=True))
model_stateless.add(TimeDistributed(Dense(units=cellunits, activation = "ELU")))
model_stateless.add(GRU(units=cellunits, return_sequences=True))
model_stateless.add(TimeDistributed(Dense(units=cellunits, activation = "ELU")))
model_stateless.add(TimeDistributed(Dense(1, activation = "sigmoid")))

model_stateless.compile(loss = 'mse', optimizer = 'adam')
#model_stateless.load_weights(bestmodel_filepath)
model_stateless.load_weights(r"pkpd_combo/QDcombo6/epochs22000_train{0}.h5".format(traind))
 #%%
subject = 8
doses_to_test = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

plt.title("Train on {0} to predict {1}".format(doses_to_train,doses_to_test))

for dosett in doses_to_test:
    dose_key = "DOSE: " + str(dosett)

    predict = model_stateless.predict(combo_to_train_data[dose_key][:,:,:3])
    true = combo_to_train_data[dose_key][:,:,3]

    plt.plot(true[subject])
    plt.plot(predict[subject])
plt.legend(['true 0.03', 'predict 0.03', 'true 0.1', 'predict 0.1', 'true 0.3', 
            'predict 0.3', 'true 1.0', 'predict 1.0', 'true 3.0', 'predict 3.0',
            'true 10.0', 'predict 10.0', 'true 30.0', 'predict 30.0'], loc='center left', bbox_to_anchor=(1, 0.7))


# =============================================================================
# plt.title("Train {0} {3} to predict {1} subject {2}".format(doses_to_use, 
#                                                             dose_key, subject, training_dosing_regimen))
# =============================================================================

# =============================================================================
# dose_key = "DOSE: 0.1"
# predict = model_stateless.predict(combo[dose_key][:,:,:3])
# true = combo[dose_key][:,:,3]
# 
# plt.plot(predict[subject])
# plt.plot(true[subject])
# 
# dose_key = "DOSE: 30.0"
# predict = model_stateless.predict(combo[dose_key][:,:,:3])
# true = combo[dose_key][:,:,3]
# 
# plt.plot(predict[subject])
# plt.plot(true[subject])
# 
# dose_key = "DOSE: 10.0"
# predict = model_stateless.predict(combo[dose_key][:,:,:3])
# true = combo[dose_key][:,:,3]
# 
# plt.plot(predict[subject])
# plt.plot(true[subject])
# 
# dose_key = "DOSE: 3.0"
# predict = model_stateless.predict(combo[dose_key][:,:,:3])
# true = combo[dose_key][:,:,3]+
# 
# plt.plot(predict[subject])
# plt.plot(true[subject])
# =============================================================================



















