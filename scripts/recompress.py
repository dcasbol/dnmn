import numpy as np
import fnmatch
import os

FILE_PAT = 'vqa/Images/**/*.jpg.npz'

print 'Start recompressing'
for root, dirnames, filenames in os.walk('vqa/Images'):
	for fn in fnmatch.filter(filenames, '*.jpg.npz'):
		full_fn = os.path.join(root, fn)
		data = np.load(full_fn)
		np.savez_compressed(full_fn, data.values()[0])
print 'Recompression finished'