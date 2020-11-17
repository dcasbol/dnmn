from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from scipy.sparse import csr_matrix
import pickle
import numpy as np
import os

MAX_BATCH_SIZE = 256
TARGET_SIZE = (320,480) # From "Learning to Reason" 2017

def get_generator(datagen, batch_size):
	return datagen.flow_from_directory(
		'CLEVR_v1.0/images', # This will go through dirs (it thinks they're classes)
		target_size=TARGET_SIZE,
		batch_size=batch_size,
		shuffle=False,
		class_mode=None
	)

def save_features(features_np, filename):
	H,W,D = features_np.shape
	sparse = csr_matrix(features_np.reshape([H,W*D]))
	with open(filename, 'wb') as fd:
		pickle.dump(sparse, fd, -1)

def get_tgt_fn(fn):
	dirname, base = os.path.split(fn)
	split_dir = dirname.split('/')
	i = split_dir.index('CLEVR_v1.0')
	tgt_dir = split_dir[:i+1] + ['conv'] + [split_dir[i+2]]
	tgt_dir = '/'.join(tgt_dir)
	tgt_fn = tgt_dir+'/'+base[:-4]+'.csr'
	if not os.path.exists(tgt_dir):
		os.makedirs(tgt_dir)
	return dirname, tgt_fn

def main():
	model = VGG16(include_top=False, weights='imagenet',
		input_shape=TARGET_SIZE + (3,))

	datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
	n_images = get_generator(datagen, 1).samples
	print(n_images, 'images found')

	# Adjust batch size, so that keras doesn't skip any sample
	# Keras will skip last samples if final batch isn't the exact size.
	BATCH_SIZE = MAX_BATCH_SIZE
	while n_images/BATCH_SIZE > n_images//BATCH_SIZE:
		BATCH_SIZE -= 1
	print('Batch size set to', BATCH_SIZE)

	img_generator = get_generator(datagen, BATCH_SIZE)
	curr_dir = ''
	last_idx = -1

	n_batches = n_images // BATCH_SIZE

	for x_batch in img_generator:
		features = model.predict(x_batch)
		idx   = ((img_generator.batch_index-1)%n_batches)*BATCH_SIZE
		fn_batch = img_generator.filepaths[idx:idx+BATCH_SIZE]
		for i, fn in enumerate(fn_batch):
			dirname, tgt_fn = get_tgt_fn(fn)
			save_features(features[i], tgt_fn)
		if dirname != curr_dir:
			curr_dir = dirname
			print('processing', curr_dir)
		if idx//100 > last_idx:
			last_idx = idx//100
			print(100*idx/n_images, '%')
		if img_generator.batch_index == 0:
			break

	print('processing finished')

if __name__ == '__main__':
	main()
