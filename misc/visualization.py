import matplotlib.pyplot as plt
from PIL import Image
from misc.util import to_numpy
from misc.constants import RAW_IMAGE_FILE

class MapVisualizer(object):

	def __init__(self, visualization_period):
		self._period = visualization_period
		assert self._period > 0
		self._count = 0
		plt.figure()
		plt.ion()
		plt.show()

	def update(self, hmaps, labels_str, input_sets, input_ids):
		self._count += 1
		if self._count % self._period != 0:
			return

		plt.clf()
		plt.suptitle(labels_str[0])

		plt.subplot(1,2,1)
		img = to_numpy(hmaps[0,0])
		im = plt.imshow(img, cmap='hot', vmin=0, vmax=1)
		plt.colorbar(im, orientation='horizontal', pad=0.05)
		plt.axis('off')

		plt.subplot(1,2,2)
		fn = RAW_IMAGE_FILE % (input_sets[0], input_sets[0], input_ids[0])
		img = np.array(Image.open(fn).resize((300,300)))
		plt.imshow(img)
		plt.axis('off')

		plt.draw()
		plt.pause(0.001)