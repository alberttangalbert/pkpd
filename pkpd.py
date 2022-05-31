import os
import utils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url


def get_data_min_max(records, device):
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	data_min, data_max = None, None
	inf = torch.tensor(float("inf")).to(device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max


class PhysioNet(object):


	def __init__(self, root, train=True, download=False,
		quantization = 0.1, n_samples = None, device = torch.device("cpu")):

		self.root = root
		self.train = train
		self.device = device
		self.reduce = "average"
		self.quantization = quantization


		


	




def variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][2].shape[1]
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	combined_tt = combined_tt.to(device)

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	
	combined_labels = None
	N_labels = 1

	combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
	combined_labels = combined_labels.to(device = device)
	
	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		tt = tt.to(device)
		vals = vals.to(device)
		mask = mask.to(device)
		if labels is not None:
			labels = labels.to(device)

		indices = inverse_indices[offset:offset + len(tt)]
		offset += len(tt)

		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask

		if labels is not None:
			combined_labels[b] = labels

	combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, 
		att_min = data_min, att_max = data_max)

	if torch.max(combined_tt) != 0.:
		combined_tt = combined_tt / torch.max(combined_tt)
		
	data_dict = {
		"data": combined_vals, 
		"time_steps": combined_tt,
		"mask": combined_mask,
		"labels": combined_labels}

	data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
	return data_dict


def variable_time_collate_fn2(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][2].shape[1]
	len_tt = [ex[1].size(0) for ex in batch]
	maxlen = np.max(len_tt)
	enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
	enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
	enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		currlen = tt.size(0)
		enc_combined_tt[b, :currlen] = tt.to(device) 
		enc_combined_vals[b, :currlen] = vals.to(device) 
		enc_combined_mask[b, :currlen] = mask.to(device) 
		
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	combined_tt = combined_tt.to(device)

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

	combined_labels = None
	N_labels = 1

	combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
	combined_labels = combined_labels.to(device = device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		tt = tt.to(device)
		vals = vals.to(device)
		mask = mask.to(device)
		if labels is not None:
			labels = labels.to(device)

		indices = inverse_indices[offset:offset + len(tt)]
		offset += len(tt)

		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask

		if labels is not None:
			combined_labels[b] = labels

	combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, 
		att_min = data_min, att_max = data_max)
	enc_combined_vals, _, _ = utils.normalize_masked_data(enc_combined_vals, enc_combined_mask, 
		att_min = data_min, att_max = data_max)

	if torch.max(combined_tt) != 0.:
		combined_tt = combined_tt / torch.max(combined_tt)
		enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)
		
	data_dict = {
				"enc_data":enc_combined_vals,
				"enc_mask":enc_combined_mask,
				"enc_time_steps":enc_combined_tt,
				"data": combined_vals, 
				"time_steps": combined_tt,
				"mask": combined_mask,
				"labels": combined_labels}

	data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
	return data_dict


def variable_time_collate_fn3(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][2].shape[1]
	len_tt = [ex[1].size(0) for ex in batch]
	maxlen = np.max(len_tt)
	enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
	enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
	enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		currlen = tt.size(0)
		enc_combined_tt[b, :currlen] = tt.to(device) 
		enc_combined_vals[b, :currlen] = vals.to(device) 
		enc_combined_mask[b, :currlen] = mask.to(device) 
		
	enc_combined_vals, _, _ = utils.normalize_masked_data(enc_combined_vals, enc_combined_mask, 
		att_min = data_min, att_max = data_max)

	if torch.max(enc_combined_tt) != 0.:
		enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)
		
	data_dict = {
				"observed_data": enc_combined_vals, 
				"observed_tp": enc_combined_tt,
				"observed_mask": enc_combined_mask}

	return data_dict



if __name__ == '__main__':
	torch.manual_seed(1991)

	dataset = PhysioNet('data/physionet', train=False, download=True)
	dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn)
	print(dataloader.__iter__().next())