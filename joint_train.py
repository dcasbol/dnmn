import time
import torch
import argparse
import torch.nn as nn
import misc.util as util
from model import NMN
from vqa import VQANMNDataset, nmn_collate_fn
from torch.utils.data import DataLoader
from misc.util import cudalize, cudalize_dict, lookahead, Logger, Chronometer
from misc.constants import *


def get_args():

	parser = argparse.ArgumentParser(description='Train NMN jointly')
	parser.add_argument('--epochs', type=int, default=1,
		help='Max. training epochs')
	parser.add_argument('--batch-size', type=int, default=512)
	parser.add_argument('--restore', action='store_true')
	parser.add_argument('--save', action='store_true')
	parser.add_argument('--suffix', type=str, default='',
		help='Add suffix to files. Useful when training others simultaneously.')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--wd', type=float, default=1e-2, help='Weight decay')
	parser.add_argument('--visualize', type=int, default=0,
		help='Visualize a masking example every N%. 0 is disabled.')
	parser.add_argument('--validate', action='store_true',
		help='Run validation every 1% of the dataset')
	return parser.parse_args()

def get_nmn_data(batch_dict):
	keys = ['features', 'question', 'length', 'yesno', 'root_inst', 'find_inst']
	return [ batch_dict[k] for k in keys ]


if __name__ == '__main__':

	args = get_args()

	SUFFIX = '' if args.suffix == '' else '-' + args.suffix
	FULL_NAME    = 'NMN' + SUFFIX
	LOG_FILENAME = FULL_NAME + '_log.json'
	PT_RESTORE   = FULL_NAME + '.pt'
	PT_NEW       = FULL_NAME + '-new.pt'

	nmn = NMN()

	dataset = VQANMNDataset()
	loader = DataLoader(dataset,
		batch_size = args.batch_size,
		collate_fn = nmn_collate_fn,
		shuffle    = True
	)

	if args.validate:
		valset = VQANMNDataset(set_names = 'val2014')
		val_loader = DataLoader(valset,
			batch_size = VAL_BATCH_SIZE,
			shuffle    = False,
			collate_fn = nmn_collate_fn
		)

	if args.restore:
		nmn.load_state_dict(torch.load(PT_RESTORE, map_location='cpu'))

	nmn = cudalize(nmn)
	opt = torch.optim.Adam(nmn.parameters(), lr=args.lr, weight_decay=args.wd)

	def cross_entropy(x, y):
		# L(x) = -y*log(x) -(1-y)*log(1-x)
		x = x[torch.arange(x.size(0)), y]
		ce = -((x+1e-10).log() + (1.-x+1e-10).log()).sum()
		return ce

	# --------------------
	# ---   Training   ---
	# --------------------
	logger = Logger()
	clock = Chronometer()
	last_perc = -1
	for epoch in range(args.epochs):
		print('Epoch ', epoch)
		for (i, batch_dict), last_iter in lookahead(enumerate(loader)):
			perc = (i*args.batch_size*100)//len(dataset)

			batch_dict = cudalize_dict(batch_dict, exclude=['find_inst'])
			nmn_data = get_nmn_data(batch_dict)

			# ---   begin timed block   ---
			clock.start()
			with torch.autograd.detect_anomaly():
				pred = nmn(*nmn_data)
				loss = cross_entropy(pred, batch_dict['label'])
				opt.zero_grad()
				loss.backward()
				opt.step()
			clock.stop()
			# ---   end timed block   ---

			if perc == last_perc: continue #and not last_iter: continue
			last_perc = perc

			mean_loss = loss.item()/batch_dict['label'].size(0)
			logger.log(
				epoch = epoch + perc/100,
				loss  = mean_loss,
				time  = clock.read()
			)

			tstr = time.strftime('%H:%M:%S', time.localtime(clock.read()))
			print('{} {: 3d}% - {}'.format(tstr, perc, mean_loss))

			if args.visualize > 0:
				assert False, 'TO DO'

			if args.validate:
				N = top1 = inset = wacc = 0

				nmn.eval()
				for batch_dict in val_loader:
					batch_dict = cudalize_dict(batch_dict, exclude=['find_inst'])
					nmn_data  = get_nmn_data(batch_dict)
					pred = nmn(*nmn_data)
					label = batch_dict['label']
					distr = batch_dict['distr']
					B = label.size(0)
					N += B
					top1  += util.top1_accuracy(pred, label) * B
					inset += util.inset_accuracy(pred, distr) * B
					wacc  += util.weighted_accuracy(pred, distr) * B
					break #if not last_iter: break
				nmn.train()

				logger.log(
					top_1    = top1/N,
					in_set   = inset/N,
					weighted = wacc/N
				)
				logger.print(exclude=['time', 'epoch'])

		if args.save:
			torch.save(nmn.state_dict(), PT_NEW)
			print('Model saved')

		logger.save(LOG_FILENAME)

	print('End of training. It took {} seconds'.format(clock.read()))
