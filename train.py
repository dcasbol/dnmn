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
from misc.util import cudalize, lookahead, Logger, Chronometer
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
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--restore', action='store_true')
	parser.add_argument('--save', action='store_true',
		help='Save the module after every epoch.')
	parser.add_argument('--suffix', type=str, default='',
		help='Add suffix to files. Useful when training others simultaneously.')
	parser.add_argument('--lr', type=float, default=1e-3,
		help='Specify learning rate')
	parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
	parser.add_argument('--competition', choices=['post', 'pre', 'softmax', 'relu-softmax'],
		default='pre',
		help='(find) Use division competition after sigmoid (post) or substraction before (pre)')
	parser.add_argument('--visualize', type=int, default=0,
		help='(find) Visualize a masking example every N%. 0 is disabled.')
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
		loss_fn = nn.BCEWithLogitsLoss if args.competition == 'pre' else nn.BCELoss
		loss_fn = loss_fn(reduction='sum')
	else:
		loss_fn =  nn.CrossEntropyLoss(reduction='sum')

	kwargs = dict(
		batch_size  = args.batch_size,
		shuffle     = True,
		num_workers = 4
	)
	if args.module == 'encoder':
		kwargs['collate_fn'] = encoder_collate_fn
	loader = DataLoader(dataset, **kwargs)

	if args.validate:
		valset = dict(
			describe = VQADescribeDataset,
			measure  = VQAMeasureDataset,
			encoder  = VQAEncoderDataset,
		)[args.module](set_names='val2014')
		kwargs = dict(collate_fn=encoder_collate_fn) if args.module == 'encoder' else {}
		val_loader = DataLoader(valset, batch_size = 100, shuffle = False, **kwargs)

	if args.restore:
		module.load_state_dict(torch.load(PT_RESTORE, map_location='cpu'))
	module = cudalize(module)

	opt = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.wd)

	if args.visualize > 0:
		vis = MapVisualizer(args.visualize)

	# --------------------
	# ---   Training   ---
	# --------------------
	logger = Logger()
	clock = Chronometer()
	last_perc = -1
	for epoch in range(args.epochs):
		print('Epoch ', epoch)
		for (i, batch_data), last_iter in lookahead(enumerate(loader)):
			perc = (i*args.batch_size*100)//len(dataset)

			# ---   begin timed block   ---
			clock.start()
			result = run_module(module, batch_data)
			output = result['output']

			loss = loss_fn(output, result['label'])
			opt.zero_grad()
			loss.backward()
			opt.step()
			clock.stop()
			# ---   end timed block   ---

			if perc == last_perc and not last_iter: continue
			last_perc = perc

			mean_loss = loss.item()/output.size(0)
			logger.log(
				epoch = epoch + perc/100,
				loss  = mean_loss,
				time  = clock.read()
			)

			tstr = time.strftime('%H:%M:%S', time.localtime(clock.read()))
			print('{} {: 3d}% - {}'.format(tstr, perc, mean_loss))

			if args.visualize > 0:
				keys   = ['hmap', 'label_str', 'input_set', 'input_id']
				values = [ result[k] for k in keys ]
				vis.update(*values)

			if args.validate:
				N = top1 = inset = wacc = 0

				module.eval()
				for batch_data in val_loader:
					result = run_module(module, batch_data)
					output = result['output'].softmax(1)
					label  = result['label']
					distr  = result['distr']
					B = label.size(0)
					N += B
					top1  += util.top1_accuracy(output, label) * B
					inset += util.inset_accuracy(output, distr) * B
					wacc  += util.weighted_accuracy(output, distr) * B
					if not last_iter: break
				module.train()
				
				logger.log(
					top_1    = top1/N,
					in_set   = inset/N,
					weighted = wacc/N
				)
				logger.print(exclude=['time', 'epoch'])

		if args.save:
			torch.save(module.state_dict(), PT_NEW)
			print('Module saved')

		logger.save(LOG_FILENAME)

	print('End of training. It took {} seconds'.format(clock.read()))
