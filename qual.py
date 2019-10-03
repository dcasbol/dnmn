import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.util as util
from torch.utils.data import DataLoader
from vqa import VQAFindDataset
from modules import Find
from misc.constants import *
from misc.util import cudalize, Logger, Chronometer, lookahead
from misc.visualization import MapVisualizer
from misc.indices import FIND_INDEX


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

class RevMask(nn.Module):

	def __init__(self):
		super(RevMask, self).__init__()
		self._classifier = nn.Linear(IMG_DEPTH, len(FIND_INDEX))
		self._loss_fn = nn.CrossEntropyLoss(reduction='sum')

	def forward(self, attended):
		B, C = attended.size()[:2]
		return self._classifier(attended)

	def loss(self, pred, instance):
		return self._loss_fn(pred, instance)

def weighted_var(features, hmap, attended, total):
	B, C = attended.size()[:2]
	attended = attended.view(B,C,1)
	var = (features-attended).pow(2)
	wvar = (var*hmap).sum(2) / (total + 1e-10)
	return wvar.mean()

def run_find(module, batch_data, metadata, dropout=0):
	if metadata:
		features, instance, label_str, input_set, input_id = batch_data
	else:
		features, instance = batch_data
	features, instance = cudalize(features, instance)
	features = F.dropout(features, p=dropout)
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

	module  = Find()
	dataset = VQAFindDataset(metadata=metadata)

	loss_fn = nn.BCELoss
	loss_fn = loss_fn(reduction='sum')

	loader = DataLoader(dataset,
		batch_size  = args.batch_size,
		shuffle     = True,
		num_workers = 4
	)

	valset = VQAFindDataset(set_names='val2014', stop=0.05, metadata=metadata)
	val_loader = DataLoader(valset, batch_size = 200, shuffle = False)

	clock = Chronometer()
	logger = Logger()
	first_epoch = 0
	if args.restore:
		logger.load(LOG_FILENAME)
		module.load_state_dict(torch.load(PT_RESTORE, map_location='cpu'))
		clock.set_t0(logger.last('time'))
		first_epoch = int(logger.last('epoch') + 0.5)
	module = cudalize(module)
	rev = cudalize(RevMask())

	params = list(module.parameters()) + list(rev.parameters())
	opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

	if args.visualize > 0:
		vis = MapVisualizer(args.visualize)


	# --------------------
	# ---   Training   ---
	# --------------------
	last_perc = -1
	yeast = 1.
	for epoch in range(first_epoch, args.epochs):
		print('Epoch ', epoch)
		N = total_loss = total_mloss = total_rloss = total_top1 = 0
		total_anti = 0
		for (i, batch_data), last_iter in lookahead(enumerate(loader)):
			perc = (i*args.batch_size*100)//len(dataset)

			# ---   begin timed block   ---
			clock.start()
			result = run_find(module, batch_data, metadata, args.dropout)
			output = result['output']

			hmap = result['hmap']
			att = attend(result['features'], hmap)
			pred = rev(att['attended'])
			loss_rev = rev.loss(pred, result['instance'])
			loss_mod = module.loss()

			# Reversed mask
			B,_,H,W = hmap.size()
			hmap_flat = hmap.view(B,-1)
			max_vals  = hmap_flat.max(1).values.view(B,1,1,1)
			hmap_inv  = max_vals - hmap
			att = attend(result['features'], hmap_inv)
			pred_inv = rev(att['attended'])
			loss_rev_inv = rev.loss(pred_inv, result['instance'])
			max_loss = torch.tensor(B*7., dtype=torch.float, device='cuda')
			loss_rev_inv = torch.min(loss_rev_inv, max_loss)

			loss = loss_rev - loss_rev_inv + yeast*loss_mod
			opt.zero_grad()
			loss.backward()
			opt.step()
			clock.stop()
			# ---   end timed block   ---

			B = output.size(0)
			N += B
			total_loss += loss.item()
			total_mloss += loss_mod.item()
			total_rloss += loss_rev.item()
			total_top1 += util.top1_accuracy(pred, result['instance']) * B
			total_anti += util.top1_accuracy(pred_inv, result['instance']) * B

			if perc == last_perc and not last_iter: continue
			last_perc = perc
			print('{: 3d}% - {} - {} - {}'.format(perc, total_loss/N, total_rloss/N, total_mloss/N))

			if args.visualize > 0:
				keys   = ['hmap', 'label_str', 'input_set', 'input_id']
				values = [ result[k] for k in keys ]
				vis.update(*values)

			if not last_iter: continue
			yeast = 0.5*abs(total_rloss/total_mloss)
			logger.log(
				yeast = yeast,
				epoch = epoch,
				loss = total_loss/N,
				mloss = total_mloss/N,
				rloss = total_rloss/N,
				time = clock.read(),
				top_1_train = total_top1/N,
				anti_train = total_anti/N
			)

			N = top1 = wvar = val_loss = 0
			anti = 0
			with torch.no_grad():
				for batch_data in val_loader:
					result = run_find(module, batch_data, metadata)
					att = attend(result['features'], result['hmap'])
					pred = rev(att['attended'])
					B = pred.size(0)
					N += B
					top1 += util.top1_accuracy(pred, result['instance']) * B
					a = [ att[k] for k in ['features_flat', 'hmap_flat', 'attended', 'total'] ]
					wvar += weighted_var(*a).item() * B
					val_loss += rev.loss(pred, result['instance']).item()

					# Reversed mask
					hmap = result['hmap']
					B,_,H,W = hmap.size()
					hmap_flat = hmap.view(B,-1)
					max_vals  = hmap_flat.max(1).values.view(B,1,1,1)
					hmap_inv  = max_vals - hmap
					att = attend(result['features'], hmap_inv)
					pred_inv = rev(att['attended'])
					anti += util.top1_accuracy(pred_inv, result['instance']) * B

			logger.log(
				top_1 = top1/N,
				wvar  = wvar/N,
				val_loss = val_loss/N,
				anti = anti/N
			)

			ploss = 'acc: {}, var: {}'.format(top1/N, wvar/N)
			print('{} - {}'.format(clock.read_str(), ploss))

			if args.save:
				torch.save(module.state_dict(), PT_NEW.format(epoch))
				print('Module saved')
			logger.save(LOG_FILENAME)

	total = clock.read()
	print('End of training. It took {} seconds'.format(total))
