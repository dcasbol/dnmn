import os
import torch
import argparse
import numpy as np
from vqa import VQANMNDataset, nmn_collate_fn
from modules import Find
from misc.util import cudalize, to_numpy, to_tens, DEVICE
from misc.util import Chronometer, attend_features, generate_hmaps
from torch.utils.data import DataLoader

def get_path(set_name, qid, cached_data='hmaps'):
	dirname = './cache/{}/{}'.format(set_name, cached_data)
	basename = '{}-{}-{}.npy'.format(set_name, cached_data, qid)
	filename = os.path.join(dirname, basename)
	return filename

def show_progress(i, total_iters):
	perc = (i*100)//total_iters
	if perc != show_progress.last:
		show_progress.last = perc
		print('\rProcessing... {: 3}%'.format(perc), end='')
	return perc
show_progress.last = -1

def generate_and_save(find, set_name, batch_data, clock):

	features = cudalize(batch_data['features'])

	clock.start()
	hmap     = generate_hmaps(find, batch_data['find_inst'], features)
	attended = attend_features(features, hmap)
	clock.stop()

	n_maps   = hmap.size(0)
	hmap     = to_numpy(hmap)
	attended = to_numpy(attended)

	for m, a, qid in zip(hmap, batch_data['question_id']):
		fn = get_path(set_name, qid)
		np.save(fn, m)
		fn = get_path(set_name, qid, cached_data='attended')
		np.save(fn, a)

	return n_maps

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Generate cache for attention maps.')
	parser.add_argument('find_module', type=str)
	parser.add_argument('--dataset', choices=['train2014', 'val2014'], default='train2014')
	parser.add_argument('--batch-size', type=int, default=256)
	args = parser.parse_args()

	assert not os.path.exists('./cache/{}'.format(args.dataset)),\
		"Please remove cache/{} dir before proceeding.".format(args.dataset)
	for name in ['hmaps', 'attended']:
		os.makedirs('./cache/{}/{}'.format(args.dataset, name))

	kwargs = dict(stop=0.2) if args.dataset == 'val2014' else {}
	dataset = VQANMNDataset(set_names=args.dataset, answers=False, **kwargs)

	find = Find()
	find.load_state_dict(torch.load(args.find_module, map_location='cpu'))
	find.eval()
	find = cudalize(find)

	clock     = Chronometer()
	raw_clock = Chronometer()
	raw_clock.start()

	n_generated = 0
	n_batches = len(dataset)//args.batch_size
	loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=nmn_collate_fn)

	for i, batch_data in enumerate(loader):
		show_progress(i, n_batches)
		n_generated += generate_and_save(find, args.dataset, batch_data, clock)

	raw_clock.stop()

	print('\nFinalized')
	print('{} maps generated, lasted {} seconds ({})'.format(
		n_generated, raw_clock.read(), raw_clock.read_str()
	))
	print('Inference time: {} ({})'.format(clock.read(), clock.read_str()))
