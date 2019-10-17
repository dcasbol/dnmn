from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from misc.util import save_features
import numpy as np
import os

MAX_BATCH_SIZE = 256
WIDTH = 448 # Sizes from 2nd article. They aren't in the NMN article.

def get_generator(datagen, batch_size):
	return datagen.flow_from_directory(
		'data/vqa/Images', # This will go through dirs (it thinks they're classes)
		target_size=(WIDTH, WIDTH),
		batch_size=batch_size,
		shuffle=False,
		class_mode=None
	)

def save_features(features_np, filename):
	sparse = csr_matrix(np.reshape(features_np, [MASK_WIDTH, MASK_WIDTH*IMG_DEPTH]))
	with open(filename, 'wb') as fd:
		pickle.dump(sparse, fd, -1)

def get_tgt_fn(fn):
	dirname, base = os.path.split(fn)
	assert dirname[-4:] == '/raw', 'not reading from raw images dir'
	tgt_dir = dirname[:-3]+'conv'
	tgt_fn = tgt_dir+'/'+base+'.npy'
	if not os.path.exists(tgt_dir):
		os.mkdir(tgt_dir)
	return dirname, tgt_fn

def main():
	model = VGG16(include_top=False, weights='imagenet',
		input_shape=(WIDTH, WIDTH, 3))

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
			print(idx/n_images, '%')
		if img_generator.batch_index == 0:
			break

	print('processing finished')

if __name__ == '__main__':
	main()