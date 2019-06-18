import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from vqatorch import VQAFindDataset
from modules import FindModule
from misc.util import max_divisor_batch_size, cudalize

SET_NAMES = 'train2014'
PT_FILENAME = 'find_module.pt'
OUT_FN_PATH = './intermediate/hmaps/{}-{}-{}.npy'

dataset = VQAFindDataset('./', SET_NAMES, filter_data=True, metadata=True)
batch_size = max_divisor_batch_size(len(dataset), 256)

loader = DataLoader(dataset, batch_size=batch_size)

find = FindModule()
find.load_state_dict(torch.load(PT_FILENAME, map_location='cpu'))
find.eval()
find = cudalize(find)

last_perc = -1
n_batches = len(dataset)//batch_size

for i, (features, target, _, target_str, input_set, input_id) in enumerate(loader):

	perc = (i*100)//n_batches
	if perc != last_perc:
		last_perc = perc
		print('\rProcessing...', perc, '%   ', end='')

	att_maps = find(features, target).detach().cpu().numpy()
	for att_map, set_name, img_id, map_c in zip(att_maps, input_set, input_id, target_str):

		fn = OUT_FN_PATH.format(set_name, img_id, map_c)
		with open(fn, 'wb') as fd:
			np.save(fd, att_map)

print('\nFinalized')