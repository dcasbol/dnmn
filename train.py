import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vqatorch import VQAFindDataset, VQADescribeDataset, VQAEncoderDataset, encoder_collate_fn
from modules import FindModule, DescribeModule, QuestionEncoder
from misc.constants import *
from misc.util import cudalize
from PIL import Image

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train a Module')
	parser.add_argument('module', choices=['find', 'describe', 'encoder'])
	parser.add_argument('--epochs', type=int, default=1,
		help='Max. training epochs')
	parser.add_argument('--batchsize', type=int, default=128)
	parser.add_argument('--restore', action='store_true')
	parser.add_argument('--save', action='store_true',
		help='Save the module after every epoch.')
	parser.add_argument('--suffix', type=str, default='',
		help='Add suffix to files. Useful when training others simultaneously.')
	parser.add_argument('--lr', type=float, default=1e-3,
		help='Specify learning rate')
	parser.add_argument('--competition', choices=['post', 'pre'], default='pre',
		help='(find) Use division competition after sigmoid (post) or substraction before (pre)')
	parser.add_argument('--visualize', type=int, default=0,
		help='(find) Select every N steps to visualize. 0 is disabled.')
	args = parser.parse_args()

	NUM_EPOCHS = args.epochs
	BATCH_SIZE = args.batchsize
	MOD_NAME = args.module
	SET_NAME = 'train2014'
	SUFFIX = '' if args.suffix == '' else '-' + args.suffix
	LOG_FILENAME = '{}_log{}.json'.format(MOD_NAME, SUFFIX)
	PT_RESTORE = '{}_module{}.pt'.format(MOD_NAME, SUFFIX)
	PT_NEW = '{}-new{}.pt'.format(MOD_NAME, SUFFIX)

	assert MOD_NAME == 'find' or args.visualize < 1, 'Only find module is subject to visualization.'

	kwargs = dict(
		batch_size  = BATCH_SIZE,
		shuffle     = True,
		num_workers = 4
	)
	if MOD_NAME == 'find':
		dataset = VQAFindDataset('./', SET_NAME, metadata=True)
		module  = FindModule(competition=args.competition)
		loss_fn = dict(
			pre  = nn.BCEWithLogitsLoss,
			post = nn.BCELoss
		)[args.competition](reduction='sum')

	elif MOD_NAME == 'describe':
		dataset  = VQADescribeDataset('./', SET_NAME)
		module   = DescribeModule()
		loss_fn  = nn.CrossEntropyLoss(reduction='sum')

	elif MOD_NAME == 'encoder':
		dataset = VQAEncoderDataset('./', SET_NAME)
		module  = QuestionEncoder()
		loss_fn = nn.CrossEntropyLoss(reduction='sum')
		kwargs['collate_fn'] = encoder_collate_fn

	loader = DataLoader(dataset, **kwargs)

	if args.restore:
		module.load_state_dict(torch.load(PT_RESTORE, map_location='cpu'))
	module = cudalize(module)

	opt = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=1e-4)

	if args.visualize > 0:
		plt.figure()
		plt.ion()
		plt.show()

	vis_count = 0
	last_perc = -1
	loss_list = list()
	for epoch in range(NUM_EPOCHS):
		print('Epoch ', epoch)
		for i, batch_data in enumerate(loader):
			perc = (i*BATCH_SIZE*100)//len(dataset)

			if MOD_NAME == 'find':
				features, instance, label_str, input_set, input_id = batch_data
				features, instance = cudalize(features, instance)
				ytrain, hmap = module(features, instance)
				label = cudalize(torch.ones_like(ytrain, dtype=torch.float))
				loss = loss_fn(ytrain, label)

			elif MOD_NAME == 'describe':
				mask, features, label = cudalize(batch_data)
				logits = module(mask, features)
				loss = loss_fn(logits, label)

			else:
				question, label = cudalize(batch_data)
				logits = module(question)
				loss = loss_fn(logits, label)

			opt.zero_grad()
			loss.backward()
			opt.step()

			if perc != last_perc:
				last_perc = perc
				loss_list.append(loss.item())
				print('{: 3d}% - {}'.format(perc, loss_list[-1]))
				vis_count += 1
				if args.visualize > 0 and vis_count%args.visualize == 0:
					plt.clf()
					plt.suptitle(label_str[0])

					plt.subplot(1,2,1)
					img = hmap.detach()[0,0].cpu().numpy()
					im = plt.imshow(img, cmap='hot', vmin=0, vmax=1)
					plt.colorbar(im, orientation='horizontal', pad=0.05)
					plt.axis('off')

					plt.subplot(1,2,2)
					fn = RAW_IMAGE_FILE % (input_set[0], input_set[0], input_id[0])
					img = np.array(Image.open(fn).resize((300,300)))
					plt.imshow(img)
					plt.axis('off')

					plt.draw()
					plt.pause(0.001)

		if args.save:
			torch.save(module.state_dict(), PT_NEW)
			print('Module saved')

	print('End of training')
	print(loss_list)
	with open(LOG_FILENAME,'w') as fd:
		json.dump(loss_list, fd)
