import os
import torch
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

	loader = DataLoader(dataset, batch_size=batch_size)

	find = Find()
	find.load_state_dict(torch.load(args.find_module, map_location='cpu'))
	find.eval()
	find = cudalize(find)

	last_perc = -1
	n_batches = len(dataset)//batch_size

	for i, (features, target, target_str, input_set, input_id) in enumerate(loader):

		perc = (i*100)//n_batches
		if perc != last_perc:
			last_perc = perc
			print('\rProcessing... {}%    '.format(perc), end='')

		features = cudalize(features)
		target   = cudalize(target)

		# Only generate non-existent maps
		pending = list()
		for i, (set_name, img_id, map_c) in enumerate(zip(input_set, input_id, target_str)):
			dirname, fn = get_paths(set_name, img_id, map_c)
			if not os.path.exists(fn):
				pending.append(i)
		if len(pending) == 0: continue
		indices  = cudalize(torch.tensor(pending))
		features = features[indices]
		target   = target[indices]

		att_maps = to_numpy(find(features, target))
		for i, att_map in enumerate(att_maps):
			set_name, img_id, map_c = [ d[indices[i]] for d in [input_set, input_id, target_str] ]
			dirname, fn = get_paths(set_name, img_id, map_c)
			if not os.path.exists(dirname):
				os.makedirs(dirname)
			with open(fn, 'wb') as fd:
				np.savez_compressed(fd, att_map)

	print('\nFinalized')