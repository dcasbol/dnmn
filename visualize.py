import argparse
import torch
from torch.utils.data import DataLoader
from vqatorch import VQAFindDataset
from modules import MLPFindModule
from misc.indices import FIND_INDEX
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

parser = argparse.ArgumentParser(description='Visualize Find Module')
parser.add_argument('--condition', type=str, default="",
	help='Condition visualizations on class.')
parser.add_argument('--softmax', action='store_true',
	help='Module was trained with softmax')
parser.add_argument('--wait', type=float, default=1.)
args = parser.parse_args()

cond = -1
if len(args.condition) > 0:
	cond = FIND_INDEX[args.condition]

SET_NAME = 'train2014'

findset = VQAFindDataset('./', SET_NAME)
loader = DataLoader(findset, batch_size=1, shuffle=False)

find = MLPFindModule(softmax=args.softmax)
find.load_state_dict(torch.load('find_module.pt', map_location='cpu'))
find.eval()

plt.figure()
plt.ion()
plt.show()

for features, label, paths in loader:

	if cond > 0 and label[0].item() != cond:
		continue

	hmap = find(features, label)

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
	time.sleep(args.wait)