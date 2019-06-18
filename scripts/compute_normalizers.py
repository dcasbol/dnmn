import os
import argparse
import numpy as np
from glob import iglob

IMAGE_PAT = "/home/david/DataSets/vqa/Images/train2014/conv/*.jpg.npz"

def compute_normalizers(fn_pattern):
	mean = np.zeros((512,), dtype=np.float32)
	mmt2 = np.zeros((512,), dtype=np.float32)
	count = 0
	for fn in iglob(fn_pattern):
		with np.load(fn) as zdata:
			assert len(zdata.keys()) == 1
			image_data = list(zdata.values())[0]
			sq_image_data = np.square(image_data)
			mean += np.sum(image_data, axis=(0,1))
			mmt2 += np.sum(sq_image_data, axis=(0,1))
			count += image_data.shape[0] * image_data.shape[1]
	assert count > 0, "glob didn't match any file"
	mean /= count
	mmt2 /= count
	var = mmt2 - np.square(mean)
	std = np.sqrt(var)

	return mean, std

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute image normalizers')
	parser.add_argument('datasetpath', type=str, default='~/DataSets/vqa',
		help='Path to VQA dataset.')
	args = parser.parse_args()

	fn_pattern = os.path.join(args.datasetpath, 'Images/train2014/conv/*.jpg.npz')
	fn_pattern = os.path.expanduser(fn_pattern)

	mean, std = compute_normalizers(fn_pattern)
	np.savez_compressed('normalizers.npz', mean=mean, std=std)