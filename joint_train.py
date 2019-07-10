import torch
import torch.nn as nn
from model import NMN
from vqa import VQANMNDataset, nmn_collate_fn
from torch.utils.data import DataLoader


if __name__ == '__main__':

	NUM_EPOCHS = 1
	nmn = NMN()

	dataset = VQANMNDataset()
	loader = DataLoader(dataset,
		batch_size = 3,
		shuffle = False,
		collate_fn = nmn_collate_fn
	)

	def loss_fn(x, y):
		return -(x[torch.arange(x.size(0)), y] + 1e-10).log().sum()
		
	opt = torch.optim.Adam(nmn.parameters(), lr=1e-3, weight_decay=1e-4)

	for epoch in range(NUM_EPOCHS):
		for i, (question, length, yesno, features, root_idx, find_ids, label, distr) in enumerate(loader):
			pred = nmn(features, question, length, yesno, root_idx, find_ids)
			loss = loss_fn(pred, label)
			opt.zero_grad()
			loss.backward()
			opt.step()
			print(i, loss.item())