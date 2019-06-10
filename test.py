import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vqatorch import VQAFindDataset
from modules import MLPFindModule
from misc.indices import ANSWER_INDEX, FIND_INDEX, UNK_ID
from misc.util import cudalize
from PIL import Image

parser = argparse.ArgumentParser(description='Train Find Module')
parser.add_argument('--epochs', type=int, default=1,
	help='Max. training epochs')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--visualize', type=int, default=0,
	help='Select every N steps to visualize. 0 is disabled.')
parser.add_argument('--restore', action='store_true')
parser.add_argument('--save', action='store_true',
	help='Save the module periodically')
parser.add_argument('--softmax', action='store_true',
	help='Use Softmax while training instead of sigmoid competition.')
args = parser.parse_args()

NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
SET_NAME = 'train2014'

findset = VQAFindDataset('./', SET_NAME)
loader = DataLoader(findset, batch_size=BATCH_SIZE, shuffle=True)

find = MLPFindModule(args.softmax)
if args.restore:
	find.load_state_dict(torch.load('find_module.pt', map_location='cpu'))
find = cudalize(find)

loss_fn = nn.BCELoss()

opt = torch.optim.Adam(find.parameters(), lr=1e-3)

if args.visualize > 0:
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

		features = cudalize(features)
		label = cudalize(label)

		batch_size = features.size(0)
		hmap = find(features, label)
		avg_pool = hmap.view(batch_size, -1).mean(1)
		ones = cudalize(torch.ones_like(avg_pool))
		loss = loss_fn(avg_pool, ones)

		opt.zero_grad()
		loss.backward()
		opt.step()

		if perc >= last_perc+0.01:
			last_perc = last_perc+0.01
			print(perc, loss.item())
			loss_list.append(loss.item())
			n += 1
			if args.visualize > 0 and n%args.visualize == 0:
				plt.clf()
				plt.suptitle(FIND_INDEX.get(label[0].item()))

				plt.subplot(1,2,1)
				img = hmap.detach()[0,0].cpu().numpy()
				plt.imshow(img, cmap='hot', vmin=0, vmax=1)
				plt.axis('off')

				plt.subplot(1,2,2)
				fn = paths[0]
				img = np.array(Image.open(fn).resize((300,300)))
				plt.imshow(img)
				plt.axis('off')

				plt.draw()
				plt.pause(0.001)

	if args.save:
		torch.save(find.state_dict(), 'find_module-new.pt')
		print('Module saved')

print(loss_list)
with open('training_log.json','w') as fd:
	json.dump(loss_list, fd)