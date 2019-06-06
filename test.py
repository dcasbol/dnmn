import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from vqatorch import VQAFindDataset
from modules import FindModule
from misc.indices import ANSWER_INDEX, MODULE_INDEX, UNK_ID
from PIL import Image
import numpy as np

NUM_EPOCHS = 3
BATCH_SIZE = 40

findset = VQAFindDataset('./', 'train2014')
loader = DataLoader(findset, batch_size=BATCH_SIZE, shuffle=False)

find = FindModule()
loss_fn = nn.BCELoss()

opt = torch.optim.Adam(find.parameters(), lr=1e-3)

last_prog = -0.01
loss_list = list()
for epoch in range(NUM_EPOCHS):
	print('Epoch ', epoch)
	for i, (features, label, label_oh) in enumerate(loader):
		prog = epoch + (i*BATCH_SIZE)/len(findset)

		batch_size = features.size(0)
		hmap = find(features, label)
		red = hmap.view(batch_size, -1).mean(1)
		loss = loss_fn(red, torch.ones_like(red))

		opt.zero_grad()
		loss.backward()
		opt.step()

		if prog >= last_prog+0.01:
			last_prog = last_prog+0.01
			print(prog, loss.item())
			loss_list.append(loss.item())

print(loss_list)
