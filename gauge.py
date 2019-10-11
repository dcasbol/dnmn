import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.util as util
from torch.utils.data import DataLoader
from vqa import VQAFindGaugeDataset
from modules import Find
from misc.constants import *
from misc.util import cudalize, Logger, Chronometer, lookahead
from misc.visualization import MapVisualizer
from misc.indices import FIND_INDEX, DESC_INDEX
from torch.distributions import Categorical


def attend(features, hmap):
	B,C,H,W = features.size()
	features = features.view(B,C,-1)
	hmap = hmap.view(B,1,-1) + 1e-10
	total = hmap.sum(2)
	attended = (hmap*features).sum(2) / total
	return dict(
		features_flat = features,
		hmap_flat = hmap,
		total = total,
		attended = attended
	)


class GaugeModule(nn.Module):

	def __init__(self):
		super(GaugeModule, self).__init__()
		self._classifier = nn.Conv2d(IMG_DEPTH, len(DESC_INDEX), 1)
		self._loss_fn = nn.CrossEntropyLoss(reduction='sum')

	def forward(self, features, hmap):
		B,C,H,W = features.size()
		features = features.view(B,C,-1)
		hmap = hmap.view(B,1,-1)
		hmap = hmap - hmap.min(2, keepdim=True).values
		total = hmap.sum(2) + 1e-10
		attended = (hmap*features).sum(2) / total
		pred = self._classifier(attended)
		return pred

	def loss(self, pred, target):
		return self._loss_fn(pred, target)


def run_find(module, batch_data, metadata=False):

	features, inst_1, inst_2, label = cudalize(*batch_data[:4])
	if metadata:
		inst_str, input_set, input_id = batch_data[4:]

	output = module[inst_1](features)

	twoinst = inst_2 > 0
	inst_2 = inst_2[twoinst]
	if inst_2.size(0) > 0:
		features_2 = features[twoinst]
		out_2 = module[inst_2](features_2)
		output[twoinst] *= out_2

	result = (features, output, label)
	if metadata:
		meta = (inst_str, input_set, input_id)
		result = result + (meta,)
	return result


def get_args():

	parser = argparse.ArgumentParser(description='Train a Module')
	parser.add_argument('--epochs', type=int, default=30,
		help='Max. training epochs')
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--save', action='store_true',
		help='Save the module after every epoch.')
	parser.add_argument('--suffix', type=str, default='',
		help='Add suffix to files. Useful when training others simultaneously.')
	parser.add_argument('--lr', type=float, default=1e-3,
		help='Specify learning rate')
	parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
	parser.add_argument('--visualize', type=int, default=0,
		help='(find) Select every N steps to visualize. 0 is disabled.')
	parser.add_argument('--dropout', type=float, default=0)
	return parser.parse_args()


if __name__ == '__main__':

	args = get_args()

	SUFFIX = '' if args.suffix == '' else '-' + args.suffix
	FULL_NAME    = 'find-qual' + SUFFIX
	LOG_FILENAME = FULL_NAME + '_log.json'
	PT_RESTORE   = FULL_NAME + '.pt'
	PT_NEW       = FULL_NAME + '-ep{:02d}-new.pt'

	metadata = args.visualize > 0

	module  = cudalize(Find(dropout=args.dropout))
	gauge   = cudalize(GaugeModule())
	dataset = VQAFindGaugeDataset(metadata=metadata)

	loader = DataLoader(dataset,
		batch_size  = args.batch_size,
		shuffle     = True,
		num_workers = 4
	)

	valset = VQAFindGaugeDataset(set_names='val2014', stop=0.2, metadata=metadata)
	val_loader = DataLoader(valset, batch_size=200, shuffle=False)

	params = list(module.parameters()) + list(gauge.parameters())
	opt = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.wd)

	if args.visualize > 0:
		vis = MapVisualizer(args.visualize)

	# --------------------
	# ---   Training   ---
	# --------------------
	clock  = Chronometer()
	logger = Logger()
	last_perc = -1
	for epoch in range(args.epochs):
		print('Epoch ', epoch)
		N = total_loss = total_top1 = 0
		for (i, batch_data), last_iter in lookahead(enumerate(loader)):
			perc = (i*args.batch_size*100)//len(dataset)

			# ---   begin timed block   ---
			clock.start()
			
			result = run_find(module, batch_data, metadata)
			features, hmap, label, meta = result

			pred = gauge(features, hmap)
			loss = gauge.loss(pred, label)

			opt.zero_grad()
			loss.backward()
			opt.step()

			clock.stop()
			# ---   end timed block   ---

			B = hmap.size(0)
			N += B
			total_loss += loss.item()
			total_top1 += util.top1_accuracy(pred, label) * B

			if perc == last_perc and not last_iter: continue
			last_perc = perc
			print('{perc: 3d}% - {loss} - {acc: 6.2f}'.format(
				perc = perc, loss = total_loss/N, acc = 100*total_top1/N
			))

			if args.visualize > 0:
				vis.update(hmap, *meta)

			if not last_iter: continue
			logger.log(
				time        = clock.read(),
				epoch       = epoch,
				loss        = total_loss/N,
				top_1_train = total_top1/N
			)

			N = top1 = val_loss = 0
			with torch.no_grad():
				for batch_data in val_loader:
					features, hmap, label = run_find(module, batch_data)
					pred = gauge(features, hmap)
					B = pred.size(0)
					N += B
					top1 += util.top1_accuracy(pred, label) * B
					val_loss += gauge.loss(pred, label).item()

			logger.log(
				top_1 = top1/N,
				val_loss = val_loss/N
			)

			print('{} - acc: {perc: 6.2f}%'.format(clock.read_str(), 100*top1/N))

			if args.save:
				torch.save(module.state_dict(), PT_NEW.format(epoch))
				print('Module saved')
			logger.save(LOG_FILENAME)

	total = clock.read()
	print('End of training. It took {} seconds'.format(total))
