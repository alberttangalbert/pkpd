import os
import time
from sklearn.metrics import mean_squared_error
from keras.regularizers import L1L2
import tensorflow
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM, GRU, Bidirectional, Dropout, RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import backend as K

num_combos = 16
doses = [0.03, 0.1, 0.3, 1.0, 3.0 , 10.0, 30.0]
num_subjects = 100
timesteps = 43
features_to_train = 3
feature_to_predict = 3

def batched(i, arr, batch_size):
    return ( arr[i * batch_size:(i + 1) * batch_size])

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
def train_combo(batch_size, num_epochs, trainX, trainY, validX, validY, save_path, combo, bestmodel_filepath):
    cellunits = 42
    start_time = time.time() / 60
    nb_cuts = len(trainX) / batch_size
    ValidationCallback = define_stateful_val_loss_class(validX, validY, batch_size, nb_cuts)
    validation = ValidationCallback()
    
    model = Sequential()
    model.add(GRU(128, batch_input_shape=(batch_size, None, features_to_train), 
                   return_sequences = True, stateful = True, kernel_initializer = "orthogonal"))
    model.add(TimeDistributed(Dense(128, activation = "ELU")))
    model.add(GRU(128, return_sequences = True,stateful = True))
    model.add(TimeDistributed(Dense(128, activation = "ELU")))
    model.add(GRU(128, return_sequences = True,stateful = True))
    model.add(Dropout(0.8))
    model.add(TimeDistributed(Dense(128, activation= "ELU")))
    model.add(TimeDistributed(Dense(1, activation = "sigmoid")))
    model.summary()
    mc = ModelCheckpoint(filepath=bestmodel_filepath, monitor='loss', mode='min', verbose=1, 
                         save_best_only=True)
            
    model.compile(optimizer = "adam", loss='mean_absolute_error', metrics=['accuracy'])
    
    history = model.fit(trainX, trainY,
                            epochs = num_epochs,
                            batch_size=batch_size,
                            shuffle=False,  
                            callbacks=[mc],
                            verbose=0)
    history.history['val_loss'] = ValidationCallback.get_val_loss(validation)
    del model
    end_time = time.time()/60
    training_time = (end_time - start_time)
    print("--- %s mins ---" % training_time)
    return history, training_time

#model.add(Dense(units=43, activation = "relu", kernel_regularizer=regularizers.l1(0.01)))
#model.add(LSTM(units=cellunits, stateful=True, return_sequences=True, kernel_initializer="orthogonal"))
#model.add(LSTM(units=cellunits, stateful=True, kernel_initializer="orthogonal"))
#model.add(TimeDistributed(Dense(units=cellunits, activation = "ELU")))
#model.add(LSTM(units=cellunits, stateful=True, return_sequences=True, kernel_initializer="orthogonal"))
#model.add(LSTM(units=cellunits, stateful=True, return_sequences=True, kernel_initializer="orthogonal"))
#model.add(LSTM(units=cellunits, stateful=True, return_sequences=True, kernel_initializer="orthogonal"))
#model.add(TimeDistributed(Dense(1, activation = "ELU")))
# =============================================================================
# model.add(GRU(128, batch_input_shape=(batch_size, timesteps, features_to_train), 
#               return_sequences=True, stateful=True))
# model.add(TimeDistributed(Dense(128, activation = "ELU")))
# model.add(GRU(128, return_sequences=True, stateful=True))
# model.add(TimeDistributed(Dense(128, activation = "ELU")))
# model.add(Dropout(0.8))
# model.add(GRU(128, return_sequences=True, stateful=True))
# model.add(TimeDistributed(Dense(128, activation = "ELU")))
# model.add(Dropout(0.8))
# model.add(GRU(128, return_sequences=True, stateful=True))
# model.add(TimeDistributed(Dense(128, activation = "ELU")))
# model.add(Dropout(0.8))
# model.add(GRU(128, return_sequences=True, stateful=True))
# model.add(TimeDistributed(Dense(128, activation = "ELU")))
# model.add(Dropout(0.8))
# model.add(GRU(128, return_sequences=True, stateful=True))
# model.add(TimeDistributed(Dense(128, activation = "ELU")))
# model.add(Dropout(0.8))
# model.add(TimeDistributed(Dense(1, activation = "sigmoid"))

# =============================================================================
# =============================================================================
# model.add(LSTM(cellunits, batch_input_shape=(batch_size, timesteps, features_to_train), 
#                return_sequences=True, stateful = True))
# model.add(LSTM(cellunits, return_sequences=True, stateful = True))
# model.add(LSTM(cellunits, return_sequences=False, stateful = True))
# model.add(Dense(cellunits, activation = "relu", kernel_regularizer= regularizers.l1(0.01)))
# model.add(RepeatVector(timesteps))
# model.add(LSTM(cellunits, return_sequences=True, stateful = True))
# model.add(LSTM(cellunits, return_sequences=True, stateful = True))
# model.add(LSTM(cellunits, return_sequences=True, stateful = True))
# model.add(TimeDistributed(Dense(1, activation=("sigmoid"))))
# =============================================================================
