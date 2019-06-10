import torch
from torch.utils.data import DataLoader
from vqatorch import VQAFindDataset
from modules import MLPFindModule
from misc.indices import MODULE_INDEX
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

SET_NAME = 'train2014'

findset = VQAFindDataset('./', SET_NAME)
loader = DataLoader(findset, batch_size=1, shuffle=False)

find = MLPFindModule()
find.load_state_dict(torch.load('find_module.pt', map_location='cpu'))
find.eval()

plt.figure()
plt.ion()
plt.show()

for features, label, paths in loader:

	hmap = find(features, label)

	plt.clf()
	plt.suptitle(MODULE_INDEX.get(label[0].item()))

	plt.subplot(1,2,1)
	img = hmap.detach()[0,0].cpu().numpy()
	plt.imshow(img, cmap='hot', vmin=0, vmax=1)
	plt.axis('off')

	plt.subplot(1,2,2)
	fn = paths[0]
	plt.imshow(mpimg.imread(fn))
	plt.axis('off')

	plt.draw()
	plt.pause(0.001)
	time.sleep(1.)