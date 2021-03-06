import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from misc.util import to_numpy
from misc.constants import RAW_IMAGE_FILE

class MapVisualizer(object):

	def __init__(self, visualization_period, vmin=None, vmax=None):
		self._period = visualization_period
		assert self._period > 0
		self._count = 0
		self._cmap = 'hot'
		self._vmin = vmin
		self._vmax = vmax
		plt.figure()
		plt.ion()
		plt.show()

	def update(self, hmaps, labels_str, input_sets, input_ids,
		save_img = False, save_map = False, ext='.tiff'):

		self._count += 1
		if self._count % self._period != 0:
			return

		plt.clf()
		plt.suptitle(labels_str[0])

		plt.subplot(1,2,1)
		img = to_numpy(hmaps[0,0])
		im = plt.imshow(img,
			cmap = self._cmap,
			vmin = np.amin(img) if self._vmin is None else self._vmin,
			vmax = np.amax(img) if self._vmax is None else self._vmax
		)
		plt.colorbar(im, orientation='horizontal', pad=0.05)
		plt.axis('off')

		if save_map:
			cmap = plt.get_cmap(self._cmap)
			simg = (cmap(img)[:,:,:3]*255).astype(np.uint8)
			Image.fromarray(simg).resize((300,300)).save('map-'+labels_str[0]+ext)

		plt.subplot(1,2,2)
		fn = RAW_IMAGE_FILE % (input_sets[0], input_sets[0], input_ids[0])
		img = Image.open(fn).resize((300,300))
		plt.imshow(np.array(img))
		plt.axis('off')

		if save_img:
			img.save(labels_str[0]+ext)

		plt.draw()
		plt.pause(0.001)
