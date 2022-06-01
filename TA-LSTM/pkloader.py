import os
import re
import torch
import numpy as np
import pandas as pd
import ujson as json
import pandas as pd
import numpy as np
import random 
from torch.utils.data import Dataset, DataLoader
from pkdataloaderfunc import read_in_data, normalize_doses, create_data_XY, random_timestep_pkremoval


def parse_rec(values, masks, evals, eval_masks, deltas):
    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()
    
    return rec
    
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


def pk_parser():
    content = [] 

    doses_to_train = [0.03]
    data = read_in_data("intensiveQD.csv")
    combo_to_train = data["Combo6: [0.1, 1.0, 0.1, 10.0]"]
    combo_to_train = normalize_doses(combo_to_train, doses_to_train)
    pkdata, _ = create_data_XY(combo_to_train, doses_to_train, 3, 4)
    sparsepkdata, masks = random_timestep_pkremoval(pkdata)
    fdeltas = parse_delta(masks, "foward")
    bdeltas = parse_delta(masks, "backward")
    for i in range(100):
        rec = {}
        #foward 
        frec = parse_rec(sparsepkdata[i], np.array([masks[j] == 1 for j in range(len(masks))])[i], 
                         pkdata[i], np.array([masks[j] == 0 for j in range(len(masks))])[i], fdeltas[i])
        rec["forward"] = frec
        #backward
        brec = parse_rec(sparsepkdata[i][::-1], np.array([masks[j] == 1 for j in range(len(masks))])[i][::-1], 
                         pkdata[i][::-1], np.array([masks[j] == 0 for j in range(len(masks))])[i][::-1], bdeltas[i])
        rec["backward"] = brec
        rec["label"] = 1
        content.append(rec)
    return content

content = pk_parser()

class MySet(Dataset):
    def __init__(self):
        super(MySet, self).__init__()
        
        self.content = pk_parser()
        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = self.content[idx]
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec
    
def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: r['values'], recs)))
        masks = torch.FloatTensor(list(map(lambda r: r['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs)))

        evals = torch.FloatTensor(list(map(lambda r: r['evals'], recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'], recs)))
        forwards = torch.FloatTensor(list(map(lambda r: r['forwards'], recs)))


        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))

    return ret_dict

def get_loader(batch_size = 10, shuffle = True):
    data_set = MySet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter

def get_data():
    return MySet()