import os
import torch
import argparse
import numpy as np
from vqa import VQAFindDataset
from modules import Find
from misc.util import max_divisor_batch_size, cudalize, to_numpy
from torch.utils.data import DataLoader

def get_paths(set_name, img_id, instance):
	dirname = './cache/hmaps/{}/{}'.format(set_name, instance)
	basename = '{}-{}-{}.npz'.format(set_name, img_id, instance)
	filename = os.path.join(dirname, basename)
	return dirname, filename

def show_progress(i, total):
	perc = (i*100)//total
	if perc != show_progress.last:
		show_progress.last = perc
		print('\rProcessing... {}%    '.format(perc), end='')
	return perc
show_progress.last = -1

def filtered_generation(find, dataset, batch_size):

	n_generated = 0
	n_samples = len(dataset)
	pending = list()

	for i in range(n_samples):

		show_progress(i, n_samples)

		target, target_str, set_name, img_id = dataset.get(i, load_features=False)

		# Only generate non-existent maps
		dirname, fn = get_paths(set_name, img_id, target_str)
		if not os.path.exists(fn):
			pending.append(i)

		if (len(pending) < batch_size and i < n_samples-1) or len(pending) == 0:
			continue

		# Build batch
		features, target, target_strs, set_names, img_ids = zip(*[ dataset[p] for p in pending ])
		features = torch.tensor(features)
		target   = torch.tensor(target)
		n_generated += generate_and_save(find, features, target, set_names, img_ids, target_strs)
		pending = list()

	return n_generated

def full_generation(find, dataset, batch_size):

	perc = -1
	n_generated = 0
	n_batches = len(dataset)//batch_size
	loader = DataLoader(dataset, batch_size=batch_size)

	for i, (features, target, target_strs, set_names, img_ids) in enumerate(loader):
		show_progress(i, n_batches)
		n_generated += generate_and_save(find, features, target, set_names, img_ids, target_strs)

	return n_generated

def generate_and_save(find, features, target, set_names, img_ids, target_strs):

	features = cudalize(features)
	target   = cudalize(target)
	att_maps = to_numpy(find(features, target))

	for att_map, set_name, img_id, target_str in zip(att_maps, set_names, img_ids, target_strs):
		dirname, fn = get_paths(set_name, img_id, target_str)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		with open(fn, 'wb') as fd:
			np.savez_compressed(fd, att_map)

	return len(att_maps)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Generate cache for attention maps.')
	parser.add_argument('find_module', type=str)
	parser.add_argument('--adjust-batch', action='store_true',
		help="Adjust batch size to avoid a smaller final batch")
	parser.add_argument('--skip-existing', action='store_true',
		help="Don't generate existing maps. Faster if there are few maps missing.")
	args = parser.parse_args()

	dataset = VQAFindDataset(filter_data=False, metadata=True)
	batch_size = max_divisor_batch_size(len(dataset), 256) if args.adjust_batch else 256
	print('Batch size set to', batch_size)

	find = Find()
	find.load_state_dict(torch.load(args.find_module, map_location='cpu'))
	find.eval()
	find = cudalize(find)

	gen = filtered_generation if args.skip_existing else full_generation
	n_generated = gen(find, dataset, batch_size)

	print('\nFinalized')
	print(n_generated, 'maps generated')
