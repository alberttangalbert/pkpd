# A toy example for clustering with 2-layer TLSTM auto-encoder
# Inci M. Baytas, 2017
# How to run: Directly run the main file: python main_AE.py

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pkdataloaderfunc as pkf

import h5py
#import hdf5storage

from T_LSTM_AE import T_LSTM_AE

# A synthetic data

Data = []
Time = []
Assignments = []
with h5py.File("Clustering_Data_1D.mat") as f:#
    for column in f['Data']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(f[column[row_number]][:])
    Data.append(row_data)
    for column in f['Time']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(f[column[row_number]][:])
    Time.append(row_data)
    for column in f['Assign']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(f[column[row_number]][:])
    Assignments.append(row_data)
#%%   
Data = []
Time = []
Assignments = []

def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]
    timesteps = np.zeros(masks.shape)
    for i in range(100):
        deltas = []
        for j in range(337):
            if j == 0:
                deltas.append(np.ones(3))
            else:
                deltas.append(np.ones(3) + (1 - masks[i][j]) * deltas[-1])
        timesteps[i] = np.array(deltas)
    return timesteps

doses_to_train = [0.03]
data = pkf.read_in_data("intensiveQD.csv")
combo_to_train = data["Combo6: [0.1, 1.0, 0.1, 10.0]"]
combo_to_train = pkf.normalize_doses(combo_to_train, doses_to_train)
pkdata, pddata = pkf.create_data_XY(combo_to_train, doses_to_train, 3, 4)
sparsepkdata, pkmasks = pkf.random_timestep_removal(pkdata)
#sparsepddata, pdmasks = pkf.random_timestep_removal(pddata)
sparsepddata = pddata * pkmasks[:,:,2].reshape(100, 337, 1)
deltas = parse_delta(pkmasks, "foward")

for i in range(100):
    pk = []
    pd = []
    t = [] 
    for j in range(337):
        if pkmasks[i, j, 2] == 1:
            pk.append(pkdata[i, j, 2])
            pd.append(pddata[i, j, 0])
            t.append(deltas[i, j - 1, 2])
    Data.append(pk)
    Time.append(t)
    Assignments.append(pd)
#%%
cell_len = len(Data)


def generate_batches(data, time, assign, index):
    batch_data = np.transpose(data[i]).reshape(42, 1, 1)
    batch_time = np.transpose(time[i]).reshape(42, 1)
    batch_assign = np.transpose(assign[i]).reshape(42, 1, 1)
    return batch_data, batch_time, batch_assign
# =========================================
#====================================
# def generate_batches(data, time, assign, index):
#     batch_data = np.transpose(data[0][index])
#     batch_time = np.transpose(time[0][index])
#     batch_assign = np.transpose(assign[0][index])
#     return batch_data, batch_time, batch_assign
# =============================================================================


# set learning parameters
learning_rate = 1e-3
ae_iters = 1
#ae_iters = 10

# set network parameters
input_dim = 1 #np.size(Data[0][0],0)
hidden_dim = 8
hidden_dim2 = 2
hidden_dim3 = 8
output_dim = hidden_dim
output_dim2 = hidden_dim2
output_dim3 = input_dim

lstm_ae = T_LSTM_AE(input_dim, output_dim, output_dim2, output_dim3, hidden_dim, hidden_dim2, hidden_dim3)

loss_ae = lstm_ae.get_reconstruction_loss()
#tf.train.AdamOptimizer() => tf.optimizers.Adam()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_ae)
#optimizer = tf.optimizers.Adam()#(learning_rate=learning_rate)#.minimize(loss_ae,var_list=[W,b])

init = tf.global_variables_initializer()
input_ = []
output_ = []
with tf.Session() as sess:
    sess.run(init)
    Loss = np.zeros(ae_iters)
    for i in range(ae_iters):
        Ll = 0
        for j in range(cell_len):
            x, t, a = generate_batches(Data, Time, Assignments, j)
            #print(x.shape, t.shape, a.shape)
            _, L = sess.run([optimizer, loss_ae], feed_dict={lstm_ae.input: x, lstm_ae.time: t})
            Ll += L
        Loss[i] = Ll# / cell_len
        print('Loss: %f' %(Loss[i]))

    assign_truth = [] 
    data_reps = []
    for c in range(cell_len):
        data = np.transpose(Data[c]).reshape(42, 1, 1)
        time = np.transpose(Time[c]).reshape(42, 1)
        assign = np.array(Assignments[c]).reshape(42, 1, 1)
        reps, cells = sess.run(lstm_ae.get_representation(), feed_dict={lstm_ae.input: data, lstm_ae.time: time})
        input_ = sess.run(lstm_ae.get_input_output(), feed_dict={lstm_ae.input: data, lstm_ae.time: time})
        #print(np.array(data_reps).shape, cells.size())
# =============================================================================
#         plt.plot(reps[:,0])
#         plt.plot(reps[:,1])
#         plt.plot(data.reshape(42))
#         plt.plot(cells[:,0])
#         plt.plot(cells[:,1])
#         plt.legend(["reps0", "reps1", "data", "cell0", "cell1"])
#         plt.savefig("result.png")
#         plt.clf()
# =============================================================================
        if c == 0:
            data_reps = reps
            assign_truth = assign
        else:
            data_reps = np.concatenate((data_reps, reps))
            assign_truth = np.concatenate((assign_truth, assign))
            
    

######## Clustering ##########################
kmeans = KMeans(n_clusters=4, random_state=0, init='k-means++').fit(data_reps)
centroid_values = kmeans.cluster_centers_

plt.figure(1)
plt.scatter(data_reps[:, 0], data_reps[:, 1], c=assign_truth, s=50, alpha=0.5)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=35)
plt.title('TLSTM')
plt.show()
#%%
plt.plot(input_[0,0,:])
