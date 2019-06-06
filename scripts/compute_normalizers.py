import numpy as np
from glob import iglob

IMAGE_PAT = "/home/david/DataSets/vqa/Images/train2014/conv/*.jpg.npz"

def compute_normalizers():
	mean = np.zeros((512,))
	mmt2 = np.zeros((512,))
	count = 0
	for fn in iglob(IMAGE_PAT):
		with np.load(fn) as zdata:
			assert len(zdata.keys()) == 1
			image_data = zdata[zdata.keys()[0]]
			sq_image_data = np.square(image_data)
			mean += np.sum(image_data, axis=(0,1))
			mmt2 += np.sum(sq_image_data, axis=(0,1))
			count += image_data.shape[0] * image_data.shape[1]
	mean /= float(count)
	mmt2 /= float(count)
	var = mmt2 - np.square(mean)
	std = np.sqrt(var)

	return mean, std

if __name__ == '__main__':
	mean, std = compute_normalizers()
	np.savez_compressed('normalizers.npz', mean=mean, std=std)