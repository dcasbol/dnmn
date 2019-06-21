import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vqatorch import VQAFindDataset
from modules import FindModule
from misc.constants import *
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
parser.add_argument('--suffix', type=str, default='',
	help='Add suffix to files. Useful when training others simultaneously.')
parser.add_argument('--lr', type=float, default=1e-4,
	help='Specify learning rate')
args = parser.parse_args()

NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
SET_NAME = 'train2014'
SUFFIX = '' if args.suffix == '' else '-' + args.suffix

findset = VQAFindDataset('./', SET_NAME, metadata=True)
loader = DataLoader(findset, batch_size=BATCH_SIZE, shuffle=True)

find = FindModule()
if args.restore:
	PT_FILENAME = 'find_module{}.pt'.format(SUFFIX)
	find.load_state_dict(torch.load(PT_FILENAME, map_location='cpu'))
find = cudalize(find)

loss_fn = nn.BCELoss(reduction='sum')

opt = torch.optim.Adam(find.parameters(), lr=args.lr, weight_decay=1e-3)

if args.visualize > 0:
	plt.figure()
	plt.ion()
	plt.show()

n = 0
last_perc = -0.01
loss_list = list()
for epoch in range(NUM_EPOCHS):
	print('Epoch ', epoch)
	for i, (features, label, label_str, input_set, input_id) in enumerate(loader):
		perc = epoch + (i*BATCH_SIZE)/len(findset)

		features = cudalize(features)
		label = cudalize(label)

		batch_size = features.size(0)
		ytrain, hmap = find(features, label)
		ones = cudalize(torch.ones_like(ytrain, dtype=torch.float))
		loss = loss_fn(ytrain, ones)

		opt.zero_grad()
		loss.backward()
		opt.step()

		if perc >= last_perc+0.01:
			last_perc = last_perc+0.01
			loss_list.append(loss.item())
			print('{: 6.2f}% - {}        '.format(perc, loss_list[-1]))
			n += 1
			if args.visualize > 0 and n%args.visualize == 0:
				plt.clf()
				plt.suptitle(label_str[0])

				plt.subplot(1,2,1)
				img = hmap.detach()[0,0].cpu().numpy()
				im = plt.imshow(img, cmap='hot', vmin=0, vmax=1)
				plt.colorbar(im, orientation='horizontal')
				plt.axis('off')

				plt.subplot(1,2,2)
				fn = RAW_IMAGE_FILE % (input_set[0], input_set[0], input_id[0])
				img = np.array(Image.open(fn).resize((300,300)))
				plt.imshow(img)
				plt.axis('off')

				plt.draw()
				plt.pause(0.001)

	if args.save:
		PT_FILENAME = 'find_module-out{}.pt'.format(SUFFIX)
		torch.save(find.state_dict(), PT_FILENAME)
		print('Module saved')

print('End of training')
print(loss_list)
LOG_FILENAME = 'training_log{}.json'.format(SUFFIX)
with open(LOG_FILENAME,'w') as fd:
	json.dump(loss_list, fd)
