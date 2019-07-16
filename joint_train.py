import torch
import torch.nn as nn
from model import NMN
from vqa import VQANMNDataset, nmn_collate_fn
from torch.utils.data import DataLoader


def get_args():

	parser = argparse.ArgumentParser(description='Train NMN jointly')
	parser.add_argument('--epochs', type=int, default=1,
		help='Max. training epochs')
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--restore', action='store_true')
	parser.add_argument('--save', action='store_true')
	parser.add_argument('--suffix', type=str, default='',
		help='Add suffix to files. Useful when training others simultaneously.')
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
	parser.add_argument('--visualize', type=int, default=0,
		help='Visualize collected data every N steps. 0 means disabled.')
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

	nmn = NMN()

	dataset = VQANMNDataset()
	loader = DataLoader(dataset,
		batch_size = args.batch_size,
		shuffle = False,
		collate_fn = nmn_collate_fn
	)

	def loss_fn(x, y):
		return -(x[torch.arange(x.size(0)), y] + 1e-10).log().sum()
		
	opt = torch.optim.Adam(nmn.parameters(), lr=1e-3, weight_decay=1e-4)

	for epoch in range(args.epochs):
		for i, (question, length, yesno, features, root_idx, find_ids, label, distr) in enumerate(loader):
			pred = nmn(features, question, length, yesno, root_idx, find_ids)
			loss = loss_fn(pred, label)
			opt.zero_grad()
			loss.backward()
			opt.step()
			print(i, loss.item())