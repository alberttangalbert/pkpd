#useful python libraries 
import numpy as np
import os
import time
import pickle
import math
#custom libraries 
from data_parse import read_in_data, create_data_XY, create_tv_XY
from train_func import train_combo
#%%
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
combo_to_test = 3
doses_to_train = [0.03, 0.1]
tvratio = 9/10

#%%
#combos is the entire data set while combo is the combo to test data
combos, combo_to_key = read_in_data("sdtab100.csv", combo_to_test)

#create folder for current combo
combo = "Combo" + str(combo_to_test) + ": " + str(combo_to_key[combo_to_test])
doses_to_train = [0.03, 0.1]
fileName = "\\combo%s" % combo_to_test
path = save_path + fileName
if not os.path.exists(path):
    os.makedirs(path)
#%%
#split data into training and validation 
combo = combos[combo_to_key[combo_to_test]]
dataX, dataY = create_data_XY(combo, doses_to_train, features_to_train, feature_to_predict)
trainX, trainY, validX, validY = create_tv_XY(dataX, dataY, tvratio)
#start training
num_epochs = 1000
batch_size = 10
dose_train = ""
for dose in doses_to_train:
    dose_train += str(dose) + "_"
dose_train = dose_train[:-1].replace(".", "")

bestmodel_filename = "bestmodel_epochs{0}_train{1}.h5".format(num_epochs, dose_train)
bestmodel_filepath = os.path.join(path, bestmodel_filename)

history = train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, 
                                     save_path, combo_to_test, dose_train, bestmodel_filepath)
#%%
def get_combos():
    return combos
