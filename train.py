import argparse
import torch
import torch.nn as nn
import misc.util as util
from torch.utils.data import DataLoader
from vqa import VQAFindDataset, VQADescribeDataset, VQAMeasureDataset
from vqa import VQAEncoderDataset, encoder_collate_fn
from modules import Find, Describe, Measure, QuestionEncoder
from misc.constants import *
from misc.util import cudalize, lookahead, Logger, Chronometer, PercentageCounter
from misc.visualization import MapVisualizer


def run_find(module, batch_data):
	features, instance, label_str, input_set, input_id = batch_data
	features, instance = cudalize(features, instance)
	output = module[instance](features)
	return dict(
		output    = output,
		hmap      = output,
		label_str = label_str,
		input_set = input_set,
		input_id  = input_id
	)

def run_describe(module, batch_data):
	mask, features, label, distr = cudalize(*batch_data[:2]+batch_data[3:])
	instance = batch_data[2]
	output = module[instance](mask, features)
	return dict(
		output = output,
		label  = label,
		distr  = distr
	)

def run_measure(module, batch_data):
	mask = cudalize(batch_data[0])
	label, distr = cudalize(*batch_data[2:])
	instance = batch_data[1]
	output = module[instance](mask)
	return dict(
		output = output,
		label  = label,
		distr  = distr
	)

def run_encoder(module, batch_data):
	question, length, label, distr = cudalize(*batch_data)
	output = module(question, length)
	return dict(
		output = output,
		label  = label,
		distr  = distr
	)

def get_args():

	parser = argparse.ArgumentParser(description='Train a Module')
	parser.add_argument('module', choices=['find', 'describe', 'measure', 'encoder'])
	parser.add_argument('--epochs', type=int, default=1,
		help='Max. training epochs')
	parser.add_argument('--batch-size', type=int, default=512)
	parser.add_argument('--restore', action='store_true')
	parser.add_argument('--save', action='store_true',
		help='Save the module after every epoch.')
	parser.add_argument('--suffix', type=str, default='',
		help='Add suffix to files. Useful when training others simultaneously.')
	parser.add_argument('--lr', type=float, default=1e-3,
		help='Specify learning rate')
	parser.add_argument('--wd', type=float, default=1e-2, help='Weight decay')
	parser.add_argument('--dropout', action='store_true')
	parser.add_argument('--competition', choices=['post', 'pre', 'softmax'],
		default='softmax',
		help='(find) Activation competition: pre/post sigmoidal or ReLU+softmax.')
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
		module  = Find(competition=args.competition, dropout=args.dropout)
		dataset = VQAFindDataset(metadata=True)
		run_module = run_find
	elif args.module == 'describe':
		module  = Describe(dropout=args.dropout)
		dataset = VQADescribeDataset()
		run_module = run_describe
	elif args.module == 'measure':
		module  = Measure(dropout=args.dropout)
		dataset = VQAMeasureDataset()
		run_module = run_measure
	else:
		module  = QuestionEncoder(dropout=args.dropout)
		dataset = VQAEncoderDataset()
		run_module = run_encoder

	if args.module == 'find':
		loss_fn = lambda a, b: module.loss()
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
		val_loader = DataLoader(valset, batch_size = VAL_BATCH_SIZE, shuffle = False, **kwargs)

	logger    = Logger()
	clock     = Chronometer()
	raw_clock = Chronometer()
	perc_cnt  = PercentageCounter(args.batch_size, len(dataset))
	first_epoch = 0

	if args.restore:
		logger.load(LOG_FILENAME)
		module.load_state_dict(torch.load(PT_RESTORE, map_location='cpu'))
		clock.set_t0(logger.last('time'))
		raw_clock.set_t0(logger.last('raw_time'))
		first_epoch = int(logger.last('epoch') + 0.5)

	module = cudalize(module)
	opt = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.wd)

	if args.visualize > 0:
		vis = MapVisualizer(args.visualize)

	# --------------------
	# ---   Training   ---
	# --------------------
	raw_clock.start()
	for epoch in range(first_epoch, args.epochs):
		print('Epoch ', epoch)
		for (i, batch_data), last_iter in lookahead(enumerate(loader)):

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

			if not perc_cnt.update(i): continue

			mean_loss = loss.item()/output.size(0)
			logger.log(
				raw_time = raw_clock.read(),
				time     = clock.read(),
				epoch    = epoch + perc_cnt.float(),
				loss     = mean_loss
			)

			raw_tstr = raw_clock.read_str()
			tstr     = clock.read_str()
			print('{}/{} {} - {}'.format(raw_tstr, tstr, perc_cnt, mean_loss))

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
					break
				module.train()
				
				logger.log(
					top_1    = top1/N,
					in_set   = inset/N,
					weighted = wacc/N
				)
				logger.print(exclude=['raw_time', 'time', 'epoch'])

		if args.save:
			torch.save(module.state_dict(), PT_NEW)
			print('Module saved')

		logger.save(LOG_FILENAME)

	print('End of training. It took {} training seconds'.format(clock.read()))
	print('{} seconds in total'.format(raw_clock.read()))
