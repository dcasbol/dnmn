import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
import misc.util as util
from torch.utils.data import DataLoader
from vqa import VQAFindDataset, VQADescribeDataset, VQAMeasureDataset
from vqa import VQAEncoderDataset, encoder_collate_fn
from modules import Find, Describe, Measure, QuestionEncoder
from misc.constants import *
from misc.util import cudalize, lookahead, Chronometer
from misc.visualization import MapVisualizer


def run_module(module, batch_data):
	result = dict()
	if isinstance(module, Find):
		features, instance, label_str, input_set, input_id = batch_data
		features, instance = cudalize(features, instance)
		output, hmap = module[instance](features)
		label = cudalize(torch.ones_like(output, dtype=torch.float))
		result = dict(hmap=hmap, label_str=label_str, input_set=input_set, input_id=input_id)
	elif isinstance(module, Describe):
		mask, features, instance, label, distr = cudalize(*batch_data)
		output = module[instance](mask, features)
	elif isinstance(module, Measure):
		mask, instance, label, distr = cudalize(*batch_data)
		output = module[instance](mask)
	else:
		question, length, label, distr = cudalize(*batch_data)
		output = module(question, length)

	result['output'] = output
	result['label']  = label
	if not isinstance(module, Find):
		result['distr'] = distr
	return result


def get_args():

	parser = argparse.ArgumentParser(description='Train a Module')
	parser.add_argument('module', choices=['find', 'describe', 'measure', 'encoder'])
	parser.add_argument('--epochs', type=int, default=1,
		help='Max. training epochs')
	parser.add_argument('--batchsize', type=int, default=128)
	parser.add_argument('--restore', action='store_true')
	parser.add_argument('--save', action='store_true',
		help='Save the module after every epoch.')
	parser.add_argument('--suffix', type=str, default='',
		help='Add suffix to files. Useful when training others simultaneously.')
	parser.add_argument('--lr', type=float, default=1e-3,
		help='Specify learning rate')
	parser.add_argument('--competition', choices=['post', 'pre'], default='pre',
		help='(find) Use division competition after sigmoid (post) or substraction before (pre)')
	parser.add_argument('--visualize', type=int, default=0,
		help='(find) Select every N steps to visualize. 0 is disabled.')
	parser.add_argument('--validate', action='store_true',
		help='Run validation every 1% of the dataset')
	return parser.parse_args()


if __name__ == '__main__':

	args = get_args()

	SUFFIX = '' if args.suffix == '' else '-' + args.suffix
	FULL_NAME    = args.module + SUFFIX
	LOG_FILENAME = FULL_NAME + '_log.json'
	PT_RESTORE   = FULL_NAME + '.pt'
	PT_NEW       = FULL_NAME + '-new.pt'

	assert args.module == 'find' or args.visualize < 1, 'Only find module is subject to visualization.'
	assert not (args.module == 'find' and args.validate), "Can't validate Find module"

	if args.module == 'find':
		module  = Find(competition=args.competition)
		dataset = VQAFindDataset(metadata=True)
	elif args.module == 'describe':
		module  = Describe()
		dataset = VQADescribeDataset()
	elif args.module == 'measure':
		module  = Measure()
		dataset = VQAMeasureDataset()
	else:
		module  = QuestionEncoder()
		dataset = VQAEncoderDataset()

	if args.module == 'find':
		loss_fn = dict(
			pre  = nn.BCEWithLogitsLoss,
			post = nn.BCELoss
		)[args.competition](reduction='sum')
	else:
		loss_fn =  nn.CrossEntropyLoss(reduction='sum')

	kwargs = dict(
		batch_size  = args.batchsize,
		shuffle     = True,
		num_workers = 4
	)
	if args.module == 'encoder':
		kwargs['collate_fn'] = encoder_collate_fn
	loader = DataLoader(dataset, **kwargs)

	log = dict(epoch = list(), loss = list(), time = list())
	if args.validate:
		for k in ['top1', 'inset', 'wacc']:
			log[k] = list()
		valset = dict(
			describe = VQADescribeDataset,
			measure  = VQAMeasureDataset,
			encoder  = VQAEncoderDataset,
		)[args.module](set_names='val2014')
		val_loader = DataLoader(valset, batch_size = 100, shuffle = False)

	if args.restore:
		module.load_state_dict(torch.load(PT_RESTORE, map_location='cpu'))
	module = cudalize(module)

	opt = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=1e-4)

	if args.visualize > 0:
		vis = MapVisualizer(args.visualize)

	# --------------------
	# ---   Training   ---
	# --------------------
	clock = Chronometer()
	last_perc = -1
	for epoch in range(args.epochs):
		print('Epoch ', epoch)
		for (i, batch_data), last_iter in lookahead(enumerate(loader)):
			perc = (i*args.batchsize*100)//len(dataset)

			result = run_module(module, batch_data)
			output = result['output']

			loss = loss_fn(output, result['label'])
			opt.zero_grad()
			loss.backward()
			opt.step()

			if perc == last_perc and not last_iter: continue

			t = clock.read()
			with clock.exclude():
				last_perc = perc
				log['epoch'].append(epoch + (i*args.batchsize)/len(dataset))
				log['loss'].append(loss.item()/output.size(0))
				log['time'].append(t)
				tstr = time.strftime('%H:%M:%S', time.localtime(t))
				print('{} {: 3d}% - {}'.format(tstr, perc, log['loss'][-1]))
				if args.visualize > 0:
					label_str, input_set, input_id = batch_data[2:]
					vis.update(hmap, label_str, input_set, input_id)

				if args.validate:
					N = top1 = inset = wacc = 0

					module.eval()
					for batch_data in val_loader:
						result = run_module(module, batch_data)
						output, label, distr = [ result[k] for k in ['output', 'label', 'distr'] ]
						output = output.softmax(1)
						B = label.size(0)
						N += B
						top1  += util.top1_accuracy(output, label) * B
						inset += util.inset_accuracy(output, distr) * B
						wacc  += util.weighted_accuracy(output, distr) * B
						if not last_iter:
							break
					module.train()
					
					log['top1'].append(top1/N)
					log['inset'].append(inset/N)
					log['wacc'].append(wacc/N)
					[ print(k, ':', vs[-1]) for k, vs in log.items() if k != 'epoch' ]

		if args.save:
			with clock.exclude():
				torch.save(module.state_dict(), PT_NEW)
				print('Module saved')

	total = clock.read()
	print('End of training. It took {} seconds'.format(total))
	with open(LOG_FILENAME,'w') as fd:
		json.dump(log, fd, indent='\t')
