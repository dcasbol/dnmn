import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from vqa import VQAFindDataset
from modules import Find
from misc.util import max_divisor_batch_size, cudalize, to_numpy

def get_paths(set_name, img_id, instance):
	dirname = './cache/hmaps/{}/{}'.format(set_name, instance)
	basename = '{}-{}-{}.npz'.format(set_name, img_id, instance)
	filename = os.path.join(dirname, basename)
	return dirname, filename

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Generate cache for attention maps.')
	parser.add_argument('find_module', type=str)
	args = parser.parse_args()

	dataset = VQAFindDataset(filter_data=False, metadata=True)
	batch_size = max_divisor_batch_size(len(dataset), 256)

	find = Find()
	find.load_state_dict(torch.load(args.find_module, map_location='cpu'))
	find.eval()
	find = cudalize(find)

	last_perc = -1
	n_samples = len(dataset)
	n_generated = 0
	pending = list()

	for i in range(n_samples):

		perc = (i*100)//n_samples
		if perc != last_perc:
			last_perc = perc
			print('\rProcessing... {}%    '.format(perc), end='')

		target, target_str, set_name, img_id = dataset.get(i, load_features=False)

		# Only generate non-existent maps
		dirname, fn = get_paths(set_name, img_id, map_c)
		if not os.path.exists(fn):
			pending.append(i)

		if len(pending) < batch_size and i < n_samples-1:
			continue

		# Build batch
		features, target, target_strs, set_names, img_ids = zip(*[ dataset[p] for p in pending ])
		features = cudalize(torch.tensor(features))
		target   = cudalize(torch.tensor(target))
		att_maps = to_numpy(find(features, target))

		# Save maps
		for att_map, instance, set_name, img_id in zip(att_maps, target_strs, set_names, img_ids):
			dirname, fn = get_paths(set_name, img_id, instance)
			if not os.path.exists(dirname):
				os.makedirs(dirname)
			with open(fn, 'wb') as fd:
				np.savez_compressed(fd, att_map)
		n_generated += len(pending)

	print('\nFinalized')
	print(n_generated, 'maps generated')