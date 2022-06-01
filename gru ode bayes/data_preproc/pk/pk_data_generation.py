# =============================================================================
# import gru_ode_bayes.pk_data_utils as pk_utils
# import pandas as pd
# import numpy as np
# pk_data = pk_utils.read_in_data("gru_ode_bayes/datasets/pk/intensiveQD.csv")
# pk_data = np.array(pk_data[0]["Combo6: [0.1, 1.0, 0.1, 10.0]"]["DOSE: 0.03"])
# newdata = np.zeros((100, 337, 5))
# mask = np.zeros((100, 337, 3))
# 
# for i in range(100):
#     newdata[i,:,0] = [i for j in range(337)]
#     d = [2, 0, 1, 3]
#     for j in range(4):
#         newdata[i,:,j + 1] = pk_data[i,:,d[j]]
#         if j == 3: 
#             pk_mask = [1 for _ in range(337)] #pk_utils.create_pk_sample_mask()
#             mask[i, :, j - 1] = pk_mask
#             newdata[i,:,j + 1] = newdata[i,:,j + 1] * pk_mask
#         if j == 0 or j == 1:
#             mask[i, :, j] = [1 for _ in range(337)]
# 
# newdata = newdata.reshape(100 * 337, 5)
# mask = mask.reshape(100 * 337, 3)
# #cov = np.array([0 for i in range(337 * 100)]).reshape(337 * 100, 1)
# #final_data = np.concatenate((newdata, mask, cov), axis=1)
# final_data = np.concatenate((newdata, mask), axis=1)
# 
# df = pd.DataFrame(final_data, columns = ['ID','Time','Value_1', 'Value_2', 'Value_3', 'Mask_1', 'Mask_2', 'Mask_3'])
# df.to_csv("gru_ode_bayes/datasets/pk/processed_pk.csv", index=False)
# =============================================================================
#%%
import gru_ode_bayes.pk_data_utils as pk_utils
import pandas as pd
import numpy as np
pk_data = pk_utils.read_in_data("gru_ode_bayes/datasets/pk/intensiveQD.csv")[0]["Combo6: [0.1, 1.0, 0.1, 10.0]"]
#pk_data = pk_utils.normalize_doses(pk_data, [0.03])
#%%
pk_data = np.array(pk_data["DOSE: 0.03"])
#%%
newdata = np.zeros((100, 337, 4))
mask = np.ones((100, 337, 2))

for i in range(100): 
    newdata[i,:,0] = [i for j in range(337)] #ID
    d = [2, 3, 1]
    for j in range(3):
        newdata[i,:,j + 1] = pk_data[i,:,d[j]]
        if j == 1: 
            # pk_mask = [1 for _ in range(337)] #pk_utils.create_pk_sample_mask()
            pk_mask = [1] + pk_utils.create_pk_sample_mask()[1:]
            mask[i, :, j - 1] = pk_mask
            newdata[i,:,j + 1] = newdata[i,:,j + 1] * pk_mask
# =============================================================================
#         if j == 0:
#             newdata[i,:,j + 1] = [i/2 for i in range(337)]
# =============================================================================
        if j == 0:
            mask[i, :, j] = [1 for _ in range(337)]

newdata = newdata.reshape(100 * 337, 4)
mask = mask.reshape(100 * 337, 2)
#cov = np.array([0 for i in range(337 * 100)]).reshape(337 * 100, 1)
#final_data = np.concatenate((newdata, mask, cov), axis=1)
final_data = np.concatenate((newdata, mask), axis=1)

df = pd.DataFrame(final_data, columns = ['ID','Time','Value_1', 'Value_2',  'Mask_1', 'Mask_2'])
df.to_csv("gru_ode_bayes/datasets/pk/processed_isample_pk.csv", index=False)