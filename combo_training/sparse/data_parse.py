import pandas as pd
import numpy as np

#will never change
num_combos = 16
doses = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
num_subjects = 200
timesteps = 43
features = 6
dose_scaling = 30

def read_in_data(filename, dosing_regimen):
    combos, combo_to_key = parse_csv(filename, dosing_regimen)
    return combos, combo_to_key

def parse_csv(filename, dosing_regimen):
    num_subjects = 200
    if filename == "normalQD.csv":
        num_subjects = 100
    combos = {} 
    #read in data
    dataSet = pd.read_csv(filename).drop(columns = ["Unnamed: 0", "RES", "WRES"])
    
    #delete uneeded rows 
    toDelIdxs = {*range(0, dataSet.shape[0], 44)}
    dataSet = dataSet.drop(toDelIdxs)
    dataSet = dataSet.reset_index(drop=True)
    
    #split into combinations 
    Grouped = dataSet.groupby("COMB", sort=False)
    temp = [Grouped.get_group(key) for key in Grouped.groups.keys()]
    for i in range(len(temp)):
        temp[i] = temp[i].reset_index(drop=True)
    
# =============================================================================
#     #add dosing frequency based on dosing regimen (QD, BID, TID)
#     dosing_time = 24
#     if dosing_regimen == "BID": dosing_time = 12
#     for i in range(num_combos):
#         dosing_frequency = temp[i]["TIME"] % dosing_time == 0
#         temp[i]["DOSE"] = temp[i]["DOSE"] * dosing_frequency
# =============================================================================
        
    #transfer data from temp to combos with keys of combinations
    combo_to_key = {}
    for i in range(num_combos):
        row0 = list(temp[i].iloc[0])
        combo = "Combo" + str(int(row0[9])) + ": " + str(row0[5:9])
        combo_to_key[i + 1] = combo
        combos[combo] = temp[i].drop(columns = ["COMB", "K", "KA", "KOUT", "I50",])
    
    #break each combo into dosages
    for combo in combos:
        Grouped = combos[combo].groupby("ID", sort=False)
        combos[combo] = [Grouped.get_group(key) for key in Grouped.groups.keys()]
        for i in range(len(combos[combo])):
            combos[combo][i] = np.array(combos[combo][i].reset_index(drop=True).drop(columns="ID"))
            combos[combo][i] = np.reshape(combos[combo][i], (num_subjects, timesteps, features))
            
    #turn [list] of np arrays into {dictionary} of np arrays for dosages
    for combo in combos:
        dosages = {}
        for i in range(len(combos[combo])):
            dose = "DOSE: " + str(combos[combo][i][:,0][0][0])
            dosages[dose] = combos[combo][i]
        combos[combo] = dosages
    return combos, combo_to_key

def normalize(data_arr, max_value, min_value):
    data_arr = (data_arr - min_value) / (max_value - min_value) # avoid saturation while close to 1
    return data_arr

def denorm(data_arr, max_value, min_value):
    data_arr = data_arr * (max_value - min_value) + min_value
    return data_arr

def normalize_combos(combos, filename):
    num_subjects = 200
    if filename == "normalQD.csv":
        num_subjects = 100
    for combo in combos:
        #find max and min values 
        minimum = np.amin(combos[combo]["DOSE: 0.03"][0,:,:], axis = 0)
        maximum = np.amax(combos[combo]["DOSE: 0.03"][0,:,:], axis = 0)
        for dose in combos[combo]:
            for i in range(num_subjects):
                minimum = [min(l1, l2) for l1, l2 in zip(minimum, np.amin(combos[combo][dose][i,:,:], axis = 0))]
                maximum = [max(l1, l2) for l1, l2 in zip(maximum, np.amax(combos[combo][dose][i,:,:], axis = 0))]
        #normalize 
        for dose in combos[combo]:
            for i in range(num_subjects):   
                combos[combo][dose][i,:,0] /= dose_scaling
                for j in range(1, 6):
                    currMin = minimum[j]
                    currMax = maximum[j]
                    combos[combo][dose][i,:,j] = normalize(combos[combo][dose][i,:,j], currMax, currMin)
    return combos

#combos need to be normalized across dosing regimens 
#i.e. to normalize combo 1 you need to normalize the combo1 of QD and BID together 
def normalize_QD_BID(QDcombos, BIDcombos):
    for combo in QDcombos:
        minimum = np.amin(QDcombos[combo]["DOSE: 0.03"][0,:,:], axis = 0)
        maximum = np.amax(QDcombos[combo]["DOSE: 0.03"][0,:,:], axis = 0)
        for dose in QDcombos[combo]:
            for i in range(num_subjects):
                minimum = [min(l1, l2) for l1, l2 in zip(minimum, np.amin(QDcombos[combo][dose][i,:,:], axis = 0))]
                maximum = [max(l1, l2) for l1, l2 in zip(maximum, np.amax(QDcombos[combo][dose][i,:,:], axis = 0))]
                minimum = [min(l1, l2) for l1, l2 in zip(minimum, np.amin(BIDcombos[combo][dose][i,:,:], axis = 0))]
                maximum = [max(l1, l2) for l1, l2 in zip(maximum, np.amax(BIDcombos[combo][dose][i,:,:], axis = 0))]
                
        for dose in QDcombos[combo]:
            for i in range(num_subjects):   
                QDcombos[combo][dose][i,:,0] /= dose_scaling
                BIDcombos[combo][dose][i,:,0] /= dose_scaling
                for j in range(1, 6):
                    currMin = minimum[j]
                    currMax = maximum[j]
                    QDcombos[combo][dose][i,:,j] = normalize(QDcombos[combo][dose][i,:,j], currMax, currMin)
                    BIDcombos[combo][dose][i,:,j] = normalize(BIDcombos[combo][dose][i,:,j], currMax, currMin)
    return QDcombos, BIDcombos

def split_dose_to_XY(combo, dose, feature_to_predict):
    dose_key = "DOSE: " + str(dose)
    dataX = combo[dose_key][:,:,:3]
    dataY = combo[dose_key][:,:,feature_to_predict].reshape(num_subjects, timesteps, 1)
    return dataX, dataY

def create_data_XY(combo, doses_to_train, features_to_train, feature_to_predict):
    dataX = np.zeros((len(doses_to_train) * num_subjects, timesteps, features_to_train))
    dataY = np.zeros((len(doses_to_train) * num_subjects, timesteps, 1))
    for i in range(len(doses_to_train)):
        doseKey = "DOSE: " + str(doses_to_train[i])
        dataX[i * num_subjects:(i + 1) * num_subjects, :,:3] = combo[doseKey][:,:,:3]
        dataY[i * num_subjects:(i + 1) * num_subjects,:,0] = combo[doseKey][:,:,feature_to_predict]
    return dataX, dataY

def create_tv_XY(dataX, dataY, tvratio):
    index_train = np.random.choice(len(dataX), int(len(dataX) * tvratio), replace=False)
    index_valid = np.setdiff1d(range(len(dataX)), index_train)

    validX = dataX[index_valid,:,:]
    validY = dataY[index_valid,:,:]
    trainX = dataX[index_train,:,:]
    trainY = dataY[index_train,:,:]
    return trainX, trainY, validX, validY
    
def creaate_random_XY(dataX, dataY, sparse_factor):
    keep = np.random.choice(len(dataX), int(len(dataX) * sparse_factor), replace=False)
    return dataX[keep,:,:], dataY[keep,:,:]