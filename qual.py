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
from misc.indices import FIND_INDEX


def attend(features, hmap):
	B,C,H,W = features.size()
	features = features.view(B,C,-1)
	hmap = hmap.view(B,1,-1)
	total = mask.sum(2)
	attended = (hmap*features).sum(2) / (hmap.sum(2) + 1e-10)
	return dict(
		features_flat = features,
		hmap_flat = hmap,
		total = total,
		attended = attended
	)

class RevMask(nn.Module):

	def __init__(self, find_module):
		super(RevMask, self).__init__()
		self._classifier = find_module._conv
		self._loss_fn = nn.CrossEntropyLoss(reduction='sum')

	def forward(self, attended):
		B, C = attended.size()[:2]
		attended = attended.view(B,C,1,1)
		return self._classifier(attended).view(B,-1)

	def loss(self, pred, instance):
		return self._loss_fn(pred, instance)

def weighted_var(features, hmap, attended, total):
	var = (features-attended).pow(2)
	wvar = (var*hmap).sum(2) / (total + 1e-10)
	return wvar

def run_find(module, batch_data, metadata):
	if metadata:
		features, instance, label_str, input_set, input_id = batch_data
	else:
		features, instance = batch_data
	features, instance = cudalize(features, instance)
	output = module[instance](features)
	result = dict(
		instance  = instance,
		features  = features,
		output    = output,
		hmap      = output
	)
	if metadata:
		result['label_str'] = label_str
		result['input_set'] = input_set
		result['input_id']  = input_id
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
	metadata = args.visualize > 0

	if args.module == 'find':
		module  = Find(competition='softmax')
		dataset = VQAFindDataset(metadata=metadata)
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
			find     = VQAFindDataset
		)[args.module](set_names='val2014', stop=0.05, metadata=metadata)
		kwargs = dict(collate_fn=encoder_collate_fn) if args.module == 'encoder' else {}
		val_loader = DataLoader(valset, batch_size = 200, shuffle = False, **kwargs)

	clock = Chronometer()
	logger = Logger()
	first_epoch = 0
	if args.restore:
		logger.load(LOG_FILENAME)
		module.load_state_dict(torch.load(PT_RESTORE, map_location='cpu'))
		clock.set_t0(logger.last('time'))
		first_epoch = int(logger.last('epoch') + 0.5)
	module = cudalize(module)
	rev = cudalize(RevMask(module))

	params = list(module.parameters()) + list(rev.parameters())
	opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

	if args.visualize > 0:
		vis = MapVisualizer(args.visualize)


	# --------------------
	# ---   Training   ---
	# --------------------
	last_perc = -1
	for epoch in range(first_epoch, args.epochs):
		print('Epoch ', epoch)
		N = total_loss = 0
		for (i, batch_data), last_iter in lookahead(enumerate(loader)):
			perc = (i*args.batch_size*100)//len(dataset)

			# ---   begin timed block   ---
			clock.start()
			result = run_find(module, batch_data, metadata)
			output = result['output']

			loss = module.loss()
			opt.zero_grad()
			loss.backward()
			opt.step()
			clock.stop()
			# ---   end timed block   ---

			B = output.size(0)
			N += B
			total_loss += loss.item()

			if perc != last_perc:
				last_perc = perc
				print('{: 3d}% - {}'.format(perc, total_loss/N))

		logger.log(
			epoch = epoch,
			loss = total_loss/N,
			time = clock.read()
		)

		N = top1 = wvar = 0
		with torch.no_grad():
			for batch_data in val_loader:
				result = run_find(module, batch_data, metadata)
				att = attend(result['features'], result['hmap'])
				pred = rev(att['attended'])
				B = pred.size(0)
				N += B
				top1 += util.top1_accuracy(pred, result['instance']) * B
				args = [ att[k] for k in ['features_flat', 'hmap_flat', 'attended', 'total'] ]
				wvar += weighted_var(*args)

		logger.log(
			top_1 = top1/N,
			wvar  = wvar/N
		)

		tstr = time.strftime('%H:%M:%S', time.localtime(clock.read()))
		ploss = 'acc: {}'.format(top1/N)
		print('{} - {}'.format(tstr, ploss))
		if args.visualize > 0:
			keys   = ['hmap', 'label_str', 'input_set', 'input_id']
			values = [ result[k] for k in keys ]
			vis.update(*values)

		if args.save:
			torch.save(module.state_dict(), PT_NEW)
			print('Module saved')
		logger.save(LOG_FILENAME)

	total = clock.read()
	print('End of training. It took {} seconds'.format(total))
