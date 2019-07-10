import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import misc.util as util
from torch.utils.data import DataLoader
from vqa import VQAFindDataset, VQADescribeDataset, VQAMeasureDataset
from vqa import VQAEncoderDataset, encoder_collate_fn
from modules import Find, Describe, Measure, QuestionEncoder
from misc.constants import *
from misc.util import cudalize
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

if __name__ == '__main__':

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
	args = parser.parse_args()

	NUM_EPOCHS = args.epochs
	BATCH_SIZE = args.batchsize
	MOD_NAME   = args.module
	SUFFIX = '' if args.suffix == '' else '-' + args.suffix
	FULL_NAME    = MOD_NAME + SUFFIX
	LOG_FILENAME = FULL_NAME + '_log.json'
	PT_RESTORE   = FULL_NAME + '.pt'
	PT_NEW       = FULL_NAME + '-new.pt'

	assert MOD_NAME == 'find' or args.visualize < 1, 'Only find module is subject to visualization.'

	if MOD_NAME == 'find':
		module  = Find(competition=args.competition)
		dataset = VQAFindDataset(metadata=True)
	elif MOD_NAME == 'describe':
		module  = Describe()
		dataset = VQADescribeDataset()
		valset  = VQADescribeDataset(set_names='val2014')
	elif MOD_NAME == 'measure':
		module  = Measure()
		dataset = VQAMeasureDataset()
		valset  = VQAMeasureDataset(set_names='val2014')
	else:
		module  = QuestionEncoder()
		dataset = VQAEncoderDataset()
		valset  = VQAEncoderDataset(set_names='val2014')

	if MOD_NAME == 'find':
		loss_fn = dict(
			pre  = nn.BCEWithLogitsLoss,
			post = nn.BCELoss
		)[args.competition](reduction='sum')
	else:
		loss_fn =  nn.CrossEntropyLoss(reduction='sum')

	kwargs = dict(
		batch_size  = BATCH_SIZE,
		shuffle     = True,
		num_workers = 4
	)
	if MOD_NAME == 'encoder':
		kwargs.update(collate_fn = encoder_collate_fn)
	loader = DataLoader(dataset, **kwargs)
	if MOD_NAME != 'find':
		kwargs['batch_size'] = 100
		val_loader = DataLoader(valset, **kwargs)

	if args.restore:
		module.load_state_dict(torch.load(PT_RESTORE, map_location='cpu'))
	module = cudalize(module)

	opt = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=1e-4)

	if args.visualize > 0:
		vis = MapVisualizer(args.visualize)

	last_perc = -1
	loss_list = list()
	for epoch in range(NUM_EPOCHS):
		print('Epoch ', epoch)
		for i, batch_data in enumerate(loader):
			perc = (i*BATCH_SIZE*100)//len(dataset)

			result = run_module(module, batch_data)
			output = result['output']

			loss = loss_fn(output, result['label'])
			opt.zero_grad()
			loss.backward()
			opt.step()

			if perc != last_perc:
				last_perc = perc
				loss_list.append([epoch + (i*BATCH_SIZE)/len(dataset), loss.item()/output.size(0)])
				print('{: 3d}% - {}'.format(perc, loss_list[-1][1]))
				if args.visualize > 0:
					label_str, input_set, input_id = batch_data[2:]
					vis.update(hmap, label_str, input_set, input_id)

				# Run validation
				if MOD_NAME != 'find':
					for batch_data in val_loader: break
					result = run_module(module, batch_data)
					output, label, distr = [ result[k] for k in ['output', 'label', 'distr'] ]
					output = output.softmax(1)
					print('top-1 acc   ', util.top1_accuracy(output, label))
					print('inset acc   ', util.inset_accuracy(output, distr))
					print('weighted acc', util.weighted_accuracy(output, distr))

		if args.save:
			torch.save(module.state_dict(), PT_NEW)
			print('Module saved')

	print('End of training')
	with open(LOG_FILENAME,'w') as fd:
		json.dump(loss_list, fd)
