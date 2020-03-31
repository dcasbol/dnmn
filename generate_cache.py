import os
import shutil
import json
import torch
import argparse
import numpy as np
from vqa import CacheDataset, cache_collate_fn
from model import Find, QuestionEncoder
from misc.constants import *
from misc.util import cudalize, to_numpy
from misc.util import Chronometer, attend_features, generate_hmaps
from torch.utils.data import DataLoader
import pickle

def get_path(set_name, qid, dtype='hmap'):
	return dict(
		hmap = CACHE_HMAP_FILE,
		att  = CACHE_ATT_FILE,
		qenc = CACHE_QENC_FILE
	)[dtype].format(set=set_name, qid=qid)

def show_progress(i, total_iters):
	perc = (i*100)//total_iters
	if perc != show_progress.last:
		show_progress.last = perc
		print('\rProcessing... {: 3}%'.format(perc), end='')
	return perc
show_progress.last = -1

def generate_and_save(modules, set_name, batch_data, clock, modular):

	with torch.no_grad():
		find, qenc = modules
		question_ids = batch_data['question_id']

		if find is not None:
			features = cudalize(batch_data['features'])
			inst = cudalize(*batch_data['find_inst'])
			inst = (inst,) if isinstance(inst, torch.Tensor) else inst

			clock.start()
			hmap     = generate_hmaps(find, inst, features, modular)
			attended = attend_features(features, hmap)
			clock.stop()

			hmap     = to_numpy(hmap)
			attended = to_numpy(attended)

			for m, a, qid in zip(hmap, attended, question_ids):
				fn = get_path(set_name, qid)
				np.save(fn, m)
				fn = get_path(set_name, qid, 'att')
				np.save(fn, a)

		if qenc is not None:
			question = cudalize(batch_data['question'])
			length   = cudalize(batch_data['length'])

			clock.start()
			pred = qenc(question, length)
			clock.stop()

			pred = to_numpy(pred)

			for p, qid in zip(pred, question_ids):
				fn = get_path(set_name, qid, 'qenc')
				np.save(fn, p)

	return len(question_ids)

class ResultObject:
	pass

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Generate cache for attention maps.')
	parser.add_argument('--find-module', type=str)
	parser.add_argument('--qenc-module', type=str)
	parser.add_argument('--dataset', choices=['train2014', 'val2014'], default='train2014')
	parser.add_argument('--batch-size', type=int, default=256)
	parser.add_argument('--overwrite', action='store_true')
	parser.add_argument('--modular', action='store_true')
	parser.add_argument('--find-config', type=str, help='Res. file for hyperparams.')
	parser.add_argument('--qenc-config', type=str, help='Res. file for hyperparams.')
	args = parser.parse_args()

	assert args.find_module is not None or args.qenc_module is not None,\
		"Missing find.pt or encoder.pt"
	if args.qenc_module is not None and not args.modular:
		raise ValueError("It's useless to generate cache for Encoder without modular flag.")

	name_list = []
	if args.find_module is not None:
		name_list += ['hmaps', 'attended']
	if args.qenc_module is not None:
		name_list += ['qenc']

	for name in name_list:
		dirname = './cache/{}/{}'.format(args.dataset, name)
		if args.overwrite:
			shutil.rmtree(dirname, ignore_errors=True)
		assert not os.path.exists(dirname),\
			"Remove {!r} or run with --overwrite flag".format(dirname)
		os.makedirs('./cache/{}/{}'.format(args.dataset, name))

	kwargs = dict(stop=0.2) if args.dataset == 'val2014' else {}
	kwargs['features']  = args.find_module is not None
	kwargs['questions'] = args.qenc_module is not None
	dataset = CacheDataset(set_names=args.dataset, **kwargs)

	ignored_params = ['learning_rate','batch_size','weight_decay','dropout','softmax_attn']
	modules = list()
	for pt_file, module_class in [(args.find_module, Find), (args.qenc_module, QuestionEncoder)]:
		m = None
		if pt_file is not None:
			kwargs = {'modular':args.modular} if module_class == Find else {}
			conf_file = args.find_config if module_class == Find else args.qenc_config
			if conf_file is not None:
				with open(conf_file, 'rb') as fd:
					res = pickle.load(fd)
				c = res.x_iters[res.best_eval]
				c = { k:v for k,v in c.items() if k not in ignored_params }
				kwargs.update(c)
			m = module_class(**kwargs)
			m.load(pt_file)
			m = cudalize(m)
			m.eval()
		modules.append(m)

	clock     = Chronometer()
	raw_clock = Chronometer()
	raw_clock.start()

	n_generated = 0
	n_batches = len(dataset)//args.batch_size
	loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=cache_collate_fn)

	for i, batch_data in enumerate(loader):
		show_progress(i, n_batches)
		n_generated += generate_and_save(modules, args.dataset, batch_data, clock, args.modular)

	raw_clock.stop()

	print('\nFinalized')
	print('{} maps generated, lasted {} seconds ({})'.format(
		n_generated, raw_clock.read(), raw_clock.read_str()
	))
	print('Inference time: {} ({})'.format(clock.read(), clock.read_str()))

	log = dict(
		dataset=args.dataset,
		find_module = args.find_module,
		qenc_module = args.qenc_module,
		time=clock.read(),
		raw_time=raw_clock.read()
	)

	fn_list = [args.find_module, args.qenc_module]
	name = ')('.join([ os.path.basename(fn)[:-3] for fn in fn_list if fn is not None ])
	with open('gen_cache-{}-({})-log.json'.format(args.dataset, name), 'w') as fd:
		json.dump(log, fd)
