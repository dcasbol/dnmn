import argparse
import json
from model import NMN, NMNPretrained
from vqa import VQANMNDataset, nmn_collate_fn
from misc.util import cudalize, cudalize_dict, max_divisor_batch_size, to_numpy
from misc.indices import ANSWER_INDEX
import torch
from torch.utils.data import DataLoader


def get_nmn_data(batch_dict):
	keys = ['features', 'question', 'length', 'yesno', 'root_inst', 'find_inst']
	return [ batch_dict[k] for k in keys ]

# TO DO: this is cheap. Put object anywhere else or rely on dict.
class ResultObject:
	pass

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="""
		Generate test results for a NMN (json file).
		User must provide either the file for the whole NMN or the file for each module.
		""")
	parser.add_argument('set_name', choices=['val2014', 'test2015'])
	parser.add_argument('--encoder')
	parser.add_argument('--find')
	parser.add_argument('--describe')
	parser.add_argument('--measure')
	parser.add_argument('--nmn')
	parser.add_argument('--output', default='results.json')
	parser.add_argument('--modular-hpo-dir', type=str)
	args = parser.parse_args()

	modules = ['encoder', 'find', 'describe', 'measure']
	modules_fn = [ getattr(args, name) for name in modules ]
	if args.modular_hpo_dir is None:
		assert args.nmn is not None or None not in modules_fn,\
			'Load NMN checkpoint or all its modules.'
	else:
		assert args.nmn is None and all([m is None for m in modules_fn]),\
			"Conflict loading modules from hpo-dir and NMN checkpoint."

	dataset = VQANMNDataset(set_names=args.set_name, answers=False)
	batch_size = max_divisor_batch_size(len(dataset), 256)
	loader = DataLoader(dataset,
		batch_size = batch_size,
		collate_fn = nmn_collate_fn,
		shuffle    = False
	)

	if args.modular_hpo_dir is not None:
		nmn = NMNPretrained(args.modular_hpo_dir)
	else:
		nmn = NMN()
		if args.nmn is not None:
			nmn.load(args.nmn)
		else:
			for name, filename in zip(modules, modules_fn):
				nmn.load_module(name, filename)
	nmn = cudalize(nmn)
	nmn.eval()

	result_list = list()

	with torch.no_grad():
		last_perc = -1
		for i, batch_data in enumerate(loader):

			perc = (i*batch_size*100)//len(dataset)
			if perc != last_perc:
				last_perc = perc
				print('\r{: 3d}%'.format(perc), end='')

			batch_data = cudalize_dict(batch_data, exclude=['question_id', 'find_inst'])
			nmn_data = get_nmn_data(batch_data)
			answers = to_numpy(nmn(*nmn_data).argmax(1))

			for qid, ans in zip(batch_data['question_id'], answers):
				result = dict(
					question_id = qid,
					answer = ANSWER_INDEX.get(ans)
				)
				result_list.append(result)

	print('\nWriting results to {!r}'.format(args.output))
	with open(args.output, 'w') as fd:
		json.dump(result_list, fd)

