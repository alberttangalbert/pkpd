import numpy as np
import matplotlib.pyplot as plt
impute = r"C:\Users\huado\Documents\Python\pkpd\BRITS-master/result/brits_i_impute.npy"
x = np.load(impute)

eval_ = r"C:\Users\huado\Documents\Python\pkpd\BRITS-master/result/brits_i_eval.npy"  
xt = np.load(eval_)

i = 0

plt.plot(x[i,:,2])

plt.plot(xt[i,:,2])

#plt.scatter([i for i in range(337)], x[i,:,2]) #%%
# =============================================================================
# from pkloader import get_data
# 
# data = get_data()
# content = data.content
# =============================================================================

     