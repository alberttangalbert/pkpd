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
import os
import time
#%%
#get current file path then create folder to output results
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
save_path = os.path.join(dir_path, "pkpd_combo")

np.random.seed(7)

#data specifications 
num_combos = 16
num_doses = 7
num_subjects = 100
timesteps = 43
train_features = 1
predict_features = 1
#%%
#create dictionary for data 
combos = {} 
#read in data
dataSet = pd.read_csv("sdtab100.csv").drop(columns = ["Unnamed: 0", "RES", "WRES"])

#delete uneeded rows 
toDelIdxs = {*range(0, dataSet.shape[0], 44)}
dataSet = dataSet.drop(toDelIdxs)
dataSet = dataSet.reset_index(drop=True)

#split into combinations 
Grouped = dataSet.groupby("COMB", sort=False)
temp = [Grouped.get_group(key) for key in Grouped.groups.keys()]
for i in range(len(temp)):
    temp[i] = temp[i].reset_index(drop=True)

#add dosing frequency, every 24 hours 
for i in range(num_combos):
    dosing_frequency = temp[i]["TIME"] % 24 == 0
    temp[i]["DOSE"] = temp[i]["DOSE"] * dosing_frequency
    
#transfer data from temp to combos with keys of combinations
combinations = []
for i in range(num_combos):
    combination = ""
    row0 = list(temp[i].iloc[0])
    combinations.append(row0[5:9])
    combination = "Combo" + str(int(row0[9])) + ": " + str(row0[5:9])
    combos[combination] = temp[i].drop(columns = ["COMB", "K", "KA", "KOUT", "I50",])

#break each combo into dosages
for combo in combos:
    Grouped = combos[combo].groupby("ID", sort=False)
    combos[combo] = [Grouped.get_group(key) for key in Grouped.groups.keys()]
    for i in range(len(combos[combo])):
        combos[combo][i] = np.array(combos[combo][i].reset_index(drop=True).drop(columns="ID"))
        combos[combo][i] = np.reshape(combos[combo][i], (100, 43, 6))
        
#turn [list] of np arrays into {dictionary} of np arrays for dosages
for combo in combos:
    dosages = {}
    for i in range(len(combos[combo])):
        dose = "DOSE: " + str(combos[combo][i][:,0][0][0])
        dosages[dose] = combos[combo][i]
    combos[combo] = dosages
#%% 
#normalize training data 
dose_scaling = 33
dosages = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
#define functions 
def normalize(data_arr, max_value, min_value):
    data_arr = (data_arr - min_value) / (max_value - min_value)#/ 1.1 # avoid saturation while close to 1
    return(data_arr)

def denorm(data_arr, max_value, min_value):
    # data_arr = data_arr * (max_value - min_value) * 1.1 + min_value
    data_arr = data_arr * (max_value - min_value) + min_value
    return(data_arr)

#create max and min dictionaries for normalizing 
minimum = {}
maximum = {}
for combo in combos:
    minimum[combo] = {}
    maximum[combo] = {}
    for dose in dosages:
        doseKey = "DOSE" + ": " + str(dose)
        minimum[combo][doseKey] = []
        maximum[combo][doseKey] = []
        for i in range(num_subjects):
            minimum[combo][doseKey].append(np.amin(combos[combo][doseKey][i,:,:], axis = 0))
            maximum[combo][doseKey].append(np.amax(combos[combo][doseKey][i,:,:], axis = 0))
            
#normalize
for combo in combos:
    for dose in dosages:
        doseKey = "DOSE" + ": " + str(dose)
        for i in range(num_subjects):           
            combos[combo][doseKey][i,:,:][:,0] = combos[combo][doseKey][i,:,:][:,0] / dose_scaling
            for j in range(1, 6):
                currMin = minimum[combo][doseKey][i][j]
                currMax = maximum[combo][doseKey][i][j]
                combos[combo][doseKey][i,:,:][:,j] = normalize(combos[combo][doseKey][i,:,:][:,j], currMax, currMin)
#%%
#organize data for testing
#define what combination  test
combos_to_test = [3]
dosages = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
#3 for PRED, 4 for DV, 5 for IPRED
feature_to_predict = 3

dataXY = {}
for combo_to_test in combos_to_test:
    combo = "Combo" + str(combo_to_test) + ": " + str(combinations[combo_to_test - 1])
    dataXY[combo] = {}
    for dose in dosages:
        doseStr = "DOSE: " + str(dose)
        dataX = np.zeros([num_subjects, timesteps, predict_features])
        dataY = np.zeros([num_subjects, timesteps, 1])
        for i in range(num_subjects):
            dataX[i,:,0] = combos[combo][doseStr][i,:,2]
            dataY[i,:,0] = combos[combo][(doseStr)][i,:,feature_to_predict]
        dataXY[combo][doseStr] = (dataX, dataY)
# =============================================================================
# #%%
# #display combos=
# for combo in dataXY.keys():
#     newname = combo[:7].replace(":", "")
#     path = save_path + newname
#     if not os.path.exists(path):
#         os.makedirs(path)      
#     comboGraph = plt.figure()
#     title = combo
#     plt.title(title)
#     for dose in dataXY[combo].keys():
#         plt.scatter(dataXY[combo][dose][0][0, :,2], dataXY[combo][dose][1][0,:,0], 
#                     label = str(dose), s = 10)
#         for i in range(43):
#             plt.annotate(i * 4, (dataXY[combo][dose][0][0,i,2], 
#                                                      dataXY[combo][dose][1][0,i,0]))
#     plt.legend()
# =============================================================================
 #%%
def batched(i, arr, batch_size):
    return (arr[i * batch_size:(i + 1) * batch_size])

def test_on_batch_stateful(model, inputs, outputs, batch_size, nb_cuts):
    nb_batches = int(len(inputs) / batch_size)
    sum_pred = 0
    for i in range(nb_batches):
        if i % nb_cuts == 0:
            model.reset_states()
        x = batched(i, inputs, batch_size)
        y = batched(i, outputs, batch_size)
        sum_pred += sum(model.test_on_batch(x, y))
    mean_pred = sum_pred / nb_batches
    return (mean_pred)

def define_stateful_val_loss_class(inputs, outputs, batch_size, nb_cuts):
    class ValidationCallback(Callback):
        def __init__(self):
            self.val_loss = []

        def on_epoch_end(self, epoch, logs={}):
            mean_pred = test_on_batch_stateful(self.model, inputs, outputs, batch_size, nb_cuts)
            print('val_loss: {:0.3e}'.format(mean_pred), end='')
            self.val_loss += [mean_pred]

        def get_val_loss(self):
            return (self.val_loss)

    return (ValidationCallback)
def train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, save_path, combo, dose_train, bestmodel_filepath):
    cellunits = 43
    start_time = time.time() / 60
    nb_cuts = len(trainX) / batch_size
    ValidationCallback = define_stateful_val_loss_class(validX, validY, batch_size, nb_cuts)
    validation = ValidationCallback()

    model = Sequential()
    model.add(LSTM(units=cellunits, stateful=True, return_sequences=True, kernel_initializer="orthogonal", 
                            batch_input_shape=(batch_size, None, predict_features)))
    model.add(TimeDistributed(Dense(units=cellunits, activation="linear"))) 
    model.add(TimeDistributed(Dense(1)))
    

    mc = ModelCheckpoint(filepath=bestmodel_filepath, monitor='loss', mode='min', verbose=1, 
                         save_best_only=True)
            
    model.compile(optimizer = "adam", loss='mean_absolute_error', metrics=['accuracy'])
    
    history = model.fit(trainX, trainY,
                            epochs = num_epochs,
                            batch_size=batch_size,
                            shuffle=False, 
                            callbacks=[mc, validation],
                            verbose=0)
    history.history['val_loss'] = ValidationCallback.get_val_loss(validation)
    del model
    end_time = time.time()/60
    training_time = (end_time - start_time)
    print("--- %s mins ---" % training_time)
    return history, training_time
#%%
num_epochs = 1000
batch_size = 10
for combo in dataXY.keys():
    count = combo[5:7].replace(":", "")
    newname = "\\combo%s"%count
    print(count)
    path = save_path + newname
    if not os.path.exists(path):
        os.makedirs(path)
    #->
    doses_to_train = [dosages[0]]
    for i in range(1, len(dosages) - 1):
        doses_to_train.append(dosages[i])
        prep_trainX = np.zeros((len(doses_to_train) * num_subjects, timesteps, predict_features))
        prep_trainY = np.zeros((len(doses_to_train) * num_subjects, timesteps, 1))
        for i in range(len(doses_to_train)):
            doseKey = "DOSE: " + str(doses_to_train[i])
            prep_trainX[i * 100:(i + 1) * 100, :, :], prep_trainY[i * 100:(i + 1) * 100, :,] = dataXY[combo][doseKey]
        
        tvratio = 9/10 # 9 different doses, for easier application
        
        index_train = np.random.choice(len(prep_trainX), int(len(prep_trainX)*tvratio), replace=False)
        index_valid = np.setdiff1d(range(len(prep_trainX)), index_train)
        
        trainX = prep_trainX[index_train,:,:]
        trainY = prep_trainY[index_train,:,:]
        validX = prep_trainX[index_valid,:,:]
        validY = prep_trainY[index_valid,:,:]

        dose_train = ""
        for dose in doses_to_train:
            dose_train += str(dose) + "_"
        dose_train = dose_train[:-1].replace(".", "")

        bestmodel_filename = "LSTMwo_epochs{0}_train{1}.h5".format(num_epochs, dose_train)
        bestmodel_filepath = os.path.join(path, bestmodel_filename)
        
        history, training_time = train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, 
                                             save_path, count, dose_train, bestmodel_filepath)
    #<-
    doses_to_train = [dosages[len(dosages) - 1]]
    for i in reversed(range(1, len(dosages) - 1)):
        doses_to_train.append(dosages[i])
        prep_trainX = np.zeros((len(doses_to_train) * num_subjects, timesteps, predict_features))
        prep_trainY = np.zeros((len(doses_to_train) * num_subjects, timesteps, 1))
        
        for i in range(len(doses_to_train)):
            doseKey = "DOSE: " + str(doses_to_train[i])
            prep_trainX[i * 100:(i + 1) * 100, :, :], prep_trainY[i * 100:(i + 1) * 100, :,] = dataXY[combo][doseKey]
        
        tvratio = 9/10 # 9 different doses, for easier application
        
        index_train = np.random.choice(len(prep_trainX), int(len(prep_trainX)*tvratio), replace=False)
        index_valid = np.setdiff1d(range(len(prep_trainX)), index_train)
        
        trainX = prep_trainX[index_train,:,:]
        trainY = prep_trainY[index_train,:,:]
        validX = prep_trainX[index_valid,:,:]
        validY = prep_trainY[index_valid,:,:]
        dose_train = ""
        for dose in doses_to_train:
            dose_train += str(dose) + "_"
        dose_train = dose_train[:-1].replace(".", "")
        
        newname = "\\combo%s"%count 
        path = save_path + newname
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        bestmodel_filename = "LSTMwo_epochs{0}_train{1}.h5".format(num_epochs, dose_train)
        bestmodel_filepath = os.path.join(path, bestmodel_filename)
        
        history, training_time = train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, 
                                             save_path, count, dose_train, bestmodel_filepath)  
        
    #---
    doses_to_train = [0.3, 1.0, 3.0]
    prep_trainX = np.zeros((len(doses_to_train) * num_subjects, timesteps, predict_features))
    prep_trainY = np.zeros((len(doses_to_train) * num_subjects, timesteps, 1))
    for i in range(len(doses_to_train)):
        doseKey = "DOSE: " + str(doses_to_train[i])
        prep_trainX[i * 100:(i + 1) * 100, :, :], prep_trainY[i * 100:(i + 1) * 100, :,] = dataXY[combo][doseKey]
    tvratio = 9/10 # 9 different doses, for easier application
        
    index_train = np.random.choice(len(prep_trainX), int(len(prep_trainX)*tvratio), replace=False)
    index_valid = np.setdiff1d(range(len(prep_trainX)), index_train)
    
    trainX = prep_trainX[index_train,:,:]
    trainY = prep_trainY[index_train,:,:]
    validX = prep_trainX[index_valid,:,:]
    validY = prep_trainY[index_valid,:,:]
    dose_train = ""
    for dose in doses_to_train:
        dose_train += str(dose) + "_"
    dose_train = dose_train[:-1].replace(".", "")
    
    newname = "\\combo%s"%count
    path = save_path #+ newname
        
    bestmodel_filename = "LSTMwo_combo{2}_epochs{0}_train{1}.h5".format(num_epochs, dose_train, count)
    bestmodel_filepath = os.path.join(path, bestmodel_filename)
        
    history, training_time = train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, 
                                                                 save_path, count, dose_train, bestmodel_filepath)
    
    #<->
    dose_order = [30.0, 0.03, 10.0, 0.1, 3.0, 0.3]
    doses_to_train = []
    for i in range(len(dose_order)):
        doses_to_train.append(dose_order[i])
        prep_trainX = np.zeros((len(doses_to_train) * num_subjects, timesteps, predict_features))
        prep_trainY = np.zeros((len(doses_to_train) * num_subjects, timesteps, 1))
        
        for i in range(len(doses_to_train)):
            doseKey = "DOSE: " + str(doses_to_train[i])
            prep_trainX[i * 100:(i + 1) * 100, :, :], prep_trainY[i * 100:(i + 1) * 100, :,] = dataXY[combo][doseKey]
        
        tvratio = 9/10 # 9 different doses, for easier application
        
        index_train = np.random.choice(len(prep_trainX), int(len(prep_trainX)*tvratio), replace=False)
        index_valid = np.setdiff1d(range(len(prep_trainX)), index_train)
        
        trainX = prep_trainX[index_train,:,:]
        trainY = prep_trainY[index_train,:,:]
        validX = prep_trainX[index_valid,:,:]
        validY = prep_trainY[index_valid,:,:]
        dose_train = ""
        for dose in doses_to_train:
            dose_train += str(dose) + "_"
        dose_train = dose_train[:-1].replace(".", "") 
            
        newname = "\\combo%s"%count
        path = save_path #+ newname
        
        bestmodel_filename = "LSTMwo_epochs{0}_train{1}".format(num_epochs, dose_train)
        bestmodel_filepath = os.path.join(path, bestmodel_filename)
        
        history, training_time = train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, 
                                             save_path, count, dose_train, bestmodel_filepath)
    count+=1
#%%
#organize dataXY into training and predicting data 
num_epochs = 1000
batch_size = 10

#->
count = 1
for combo in combos:
    newname = "\\combo%s"%count
    path = save_path + newname
    if not os.path.exists(path):
        os.makedirs(path)
        
    doses_to_train = [dosages[0]]
    for i in range(1, len(dosages) - 1):
        doses_to_train.append(dosages[i])
        prep_trainX = np.zeros((len(doses_to_train) * num_subjects, timesteps, predict_features))
        prep_trainY = np.zeros((len(doses_to_train) * num_subjects, timesteps, 1))
        
        for i in range(len(doses_to_train)):
            doseKey = "DOSE: " + str(doses_to_train[i])
            prep_trainX[i * 100:(i + 1) * 100, :, :], prep_trainY[i * 100:(i + 1) * 100, :,] = dataXY[doseKey]
        
        tvratio = 9/10 # 9 different doses, for easier application
        
        index_train = np.random.choice(len(prep_trainX), int(len(prep_trainX)*tvratio), replace=False)
        index_valid = np.setdiff1d(range(len(prep_trainX)), index_train)
        
        trainX = prep_trainX[index_train,:,:]
        trainY = prep_trainY[index_train,:,:]
        validX = prep_trainX[index_valid,:,:]
        validY = prep_trainY[index_valid,:,:]
        
        dose_train = ""
        for dose in doses_to_train:
            dose_train += str(dose) + "_"
        dose_train = dose_train[:-1].replace(".", "")
            
        bestmodel_filename = "bestmodel_epochs{0}_train{1}".format(num_epochs, dose_train)
        bestmodel_filepath = os.path.join(path, bestmodel_filename)
        
        history, training_time = train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, 
                                             save_path, count, dose_train, bestmodel_filepath)
    
    count+=1
    #%%
#<-
count = 1
for combo in combos:
    doses_to_train = [dosages[len(dosages) - 1]]
    newname = "\\combo%s"%count
    path = save_path + newname
    
    if not os.path.exists(path):
        os.makedirs(path)
    for i in reversed(range(1, len(dosages) - 1)):
        doses_to_train.append(dosages[i])
        prep_trainX = np.zeros((len(doses_to_train) * num_subjects, timesteps, predict_features))
        prep_trainY = np.zeros((len(doses_to_train) * num_subjects, timesteps, 1))
        
        for i in range(len(doses_to_train)):
            doseKey = "DOSE: " + str(doses_to_train[i])
            prep_trainX[i * 100:(i + 1) * 100, :, :], prep_trainY[i * 100:(i + 1) * 100, :,] = dataXY[doseKey]
        
        tvratio = 9/10 # 9 different doses, for easier application
        
        index_train = np.random.choice(len(prep_trainX), int(len(prep_trainX)*tvratio), replace=False)
        index_valid = np.setdiff1d(range(len(prep_trainX)), index_train)
        
        trainX = prep_trainX[index_train,:,:]
        trainY = prep_trainY[index_train,:,:]
        validX = prep_trainX[index_valid,:,:]
        validY = prep_trainY[index_valid,:,:]
        dose_train = ""
        for dose in doses_to_train:
            dose_train += str(dose) + "_"
        dose_train = dose_train[:-1].replace(".", "")
        
        newname = "\\combo%s"%count 
        path = save_path + newname
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        bestmodel_filename = "bestmodel_epochs{0}_train{1}".format(num_epochs, dose_train)
        bestmodel_filepath = os.path.join(path, bestmodel_filename)
        
        history, training_time = train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, 
                                                                     save_path, count, dose_train, bestmodel_filepath)
    count+=1
#---
#middle
count = 1
for combo in combos:
    
    newname = "\\combo%s"%count
    path = save_path + newname
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    doses_to_train = [0.3, 1.0, 3.0]
    prep_trainX = np.zeros((len(doses_to_train) * num_subjects, timesteps, predict_features))
    prep_trainY = np.zeros((len(doses_to_train) * num_subjects, timesteps, 1))
    
    for i in range(len(doses_to_train)):
        doseKey = "DOSE: " + str(doses_to_train[i])
        prep_trainX[i * 100:(i + 1) * 100, :, :], prep_trainY[i * 100:(i + 1) * 100, :,] = dataXY[doseKey]
    tvratio = 9/10 # 9 different doses, for easier application
        
    index_train = np.random.choice(len(prep_trainX), int(len(prep_trainX)*tvratio), replace=False)
    index_valid = np.setdiff1d(range(len(prep_trainX)), index_train)
    
    trainX = prep_trainX[index_train,:,:]
    trainY = prep_trainY[index_train,:,:]
    validX = prep_trainX[index_valid,:,:]
    validY = prep_trainY[index_valid,:,:]
    dose_train = ""
    for dose in doses_to_train:
        dose_train += str(dose) + "_"
    dose_train = dose_train[:-1].replace(".", "")
    
    newname = "\\combo%s"%count
    path = save_path #+ newname
        
    bestmodel_filename = "bestmodel_combo{2}_epochs{0}_train{1}".format(num_epochs, dose_train, count)
    bestmodel_filepath = os.path.join(path, bestmodel_filename)
        
    history, training_time = train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, 
                                                                 save_path, count, dose_train, bestmodel_filepath)
    count+=1
#<-->
count = 1
for combo in combos:
    dose_order = [30.0, 0.03, 10.0, 0.1, 3.0, 0.3]
    doses_to_train = []
    newname = "\\combo%s"%count
    path = save_path + newname
    
    if not os.path.exists(path):
            os.makedirs(path)
    for i in range(len(dose_order)):
        doses_to_train.append(dose_order[i])
        prep_trainX = np.zeros((len(doses_to_train) * num_subjects, timesteps, predict_features))
        prep_trainY = np.zeros((len(doses_to_train) * num_subjects, timesteps, 1))
        
        for i in range(len(doses_to_train)):
            doseKey = "DOSE: " + str(doses_to_train[i])
            prep_trainX[i * 100:(i + 1) * 100, :, :], prep_trainY[i * 100:(i + 1) * 100, :,] = dataXY[doseKey]
        
        tvratio = 9/10 # 9 different doses, for easier application
        
        index_train = np.random.choice(len(prep_trainX), int(len(prep_trainX)*tvratio), replace=False)
        index_valid = np.setdiff1d(range(len(prep_trainX)), index_train)
        
        trainX = prep_trainX[index_train,:,:]
        trainY = prep_trainY[index_train,:,:]
        validX = prep_trainX[index_valid,:,:]
        validY = prep_trainY[index_valid,:,:]
        dose_train = ""
        for dose in doses_to_train:
            dose_train += str(dose) + "_"
        dose_train = dose_train[:-1].replace(".", "") 
            
        newname = "\\combo%s"%count
        path = save_path #+ newname
        
        bestmodel_filename = "bestmodel_epochs{0}_train{1}".format(num_epochs, dose_train)
        bestmodel_filepath = os.path.join(path, bestmodel_filename)
        
        history, training_time = train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, 
                                                                     save_path, count, dose_train, bestmodel_filepath)
    count+=1
#%%
cellunits = 43
model_stateless = Sequential()
model_stateless.add(GRU(units=cellunits, return_sequences=True, 
                        input_shape=[None, predict_features]))
# =============================================================================
# model_stateless.add(TimeDistributed(Dense(units=cellunits, activation="linear")))
# model_stateless.add(GRU(units=cellunits, return_sequences=True))
# =============================================================================
model_stateless.add(TimeDistributed(Dense(units=cellunits, activation="linear")))
model_stateless.add(TimeDistributed(Dense(1)))

model_stateless.compile(loss = 'mse', optimizer = 'adam')

bestmodel_filepath = r"C:\Users\alber\Code\Python\PKPD\pkpd_combo\combo2\bestmodel_epochs1000_train003_01_03.h5"
model_stateless.load_weights(bestmodel_filepath)
#%%
predict = model_stateless.predict(dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][0])
fig11 = plt.figure()
title = "Combo3 - 0.03, 0.1, 0.3 to predict 1 GRU, 1 Dense"
plt.title(title)
predictX = denorm(dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][0], 
                  maximum["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][0][2],
                  minimum["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][0][2])
predictY = denorm(predict[0,:,], maximum["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][0][3], 
                 minimum["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][0][3])
plt.plot(predictX[0], predictY)
true = denorm(dataXY["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][1], 
              maximum["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][0][3],
              minimum["Combo3: [0.1, 0.1, 1.0, 1.0]"]["DOSE: 1.0"][0][3])
plt.plot(true[0])
plt.legend(['predict', 'actual'], loc='upper left')
#%%
trainPredict = model_stateless.predict(dataX2)
validPredict = model_stateless.predict(validatingX)
plt.plot(trainPredict[20])
plt.plot(dataY3[20])
plt.legend(['predict', 'actual'], loc='upper left')
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




















