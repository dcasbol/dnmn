import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from vqatorch import VQAFindDataset
from modules import MLPFindModule
from misc.indices import ANSWER_INDEX, MODULE_INDEX, UNK_ID
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

NUM_EPOCHS = 3
BATCH_SIZE = 16
SET_NAME = 'train2014'

findset = VQAFindDataset('./', SET_NAME)
loader = DataLoader(findset, batch_size=BATCH_SIZE, shuffle=True)

find = MLPFindModule()
loss_fn = nn.BCELoss()

opt = torch.optim.Adam(find.parameters(), lr=1e-4)

plt.figure()
plt.ion()
plt.show()

n = 0
last_perc = -0.01
loss_list = list()
for epoch in range(NUM_EPOCHS):
	print('Epoch ', epoch)
	for i, (features, label, paths) in enumerate(loader):
		perc = epoch + (i*BATCH_SIZE)/len(findset)

		batch_size = features.size(0)
		hmap = find(features, label)
		avg_pool = hmap.view(batch_size, -1).mean(1)
		loss = loss_fn(avg_pool, torch.ones_like(avg_pool))

		opt.zero_grad()
		loss.backward()
		opt.step()

		if perc >= last_perc+0.01:
			last_perc = last_perc+0.01
			print(perc, loss.item())
			loss_list.append(loss.item())
			n += 1
			if n%2 == 0:
				plt.clf()
				plt.suptitle(MODULE_INDEX.get(label[0].item()))

				plt.subplot(1,2,1)
				img = hmap.detach()[0,0].numpy()
				plt.imshow(img, cmap='hot', vmin=0, vmax=1)
				plt.axis('off')

				plt.subplot(1,2,2)
				fn = paths[0]
				plt.imshow(mpimg.imread(fn))
				plt.axis('off')

				plt.draw()
				plt.pause(0.001)

print(loss_list)
