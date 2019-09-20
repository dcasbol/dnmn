import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vqa import VQAFindDataset
from modules import Find
from misc.indices import FIND_INDEX
from misc.constants import *
from PIL import Image

parser = argparse.ArgumentParser(description='Visualize Find Module')
parser.add_argument('ptfile', type=str,
	help='Path to .pt file storing module weights')
parser.add_argument('condition', type=str, nargs='?', default="",
	help='Condition visualizations on class.')
parser.add_argument('--wait', type=float, default=1.)
parser.add_argument('--shuffle', action='store_true')
args = parser.parse_args()

cond = -1
if len(args.condition) > 0:
	cond = FIND_INDEX[args.condition]

findset = VQAFindDataset(metadata=True)
loader = DataLoader(findset, batch_size=1, shuffle=args.shuffle)

find = Find()
find.load_state_dict(torch.load(args.ptfile, map_location='cpu'))
find.eval()

plt.figure()
plt.ion()
plt.show()

for features, instance, instance_str, input_set, input_id in loader:

	if cond > 0 and instance[0].item() != cond:
		continue

	hmap = find[instance](features)

	plt.clf()
	plt.suptitle(instance_str[0])

	ax = plt.subplot(1,2,1)
	img = hmap.detach()[0,0].cpu().numpy()
	im = plt.imshow(img, cmap='hot')
	plt.colorbar(im, orientation='horizontal', pad=0.05)
	plt.axis('off')

	plt.subplot(1,2,2)
	fn = RAW_IMAGE_FILE % (input_set[0], input_set[0], input_id[0])
	img = np.array(Image.open(fn).resize((300,300)))
	plt.imshow(img)
	plt.axis('off')

	plt.draw()
	plt.pause(0.001)
	time.sleep(args.wait)
