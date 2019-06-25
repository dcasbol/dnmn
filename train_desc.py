import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vqatorch import VQADescribeDataset
from modules import DescribeModule
from misc.constants import *
from misc.util import cudalize
from PIL import Image

parser = argparse.ArgumentParser(description='Train Describe Module')
parser.add_argument('--epochs', type=int, default=1,
	help='Max. training epochs')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--restore', action='store_true')
parser.add_argument('--save', action='store_true',
	help='Save the module periodically')
parser.add_argument('--suffix', type=str, default='',
	help='Add suffix to files. Useful when training others simultaneously.')
parser.add_argument('--lr', type=float, default=1e-3,
	help='Specify learning rate')
args = parser.parse_args()

NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
MOD_NAME = 'describe'
SET_NAME = 'train2014'
SUFFIX = '' if args.suffix == '' else '-' + args.suffix
LOG_FILENAME = '{}_log{}.json'.format(MOD_NAME, SUFFIX)
PT_RESTORE = '{}_module{}.pt'.format(MOD_NAME, SUFFIX)
PT_NEW = '{}-new{}.pt'.format(MOD_NAME, SUFFIX)

descset = VQADescribeDataset('./', SET_NAME)
loader = DataLoader(descset, batch_size=BATCH_SIZE, shuffle=True)

desc = DescribeModule()
if args.restore:
	desc.load_state_dict(torch.load(PT_RESTORE, map_location='cpu'))
desc = cudalize(desc)

loss_fn = nn.KLDivLoss(reduction='sum')
logsm_fn = nn.LogSoftmax()

opt = torch.optim.Adam(desc.parameters(), lr=args.lr, weight_decay=1e-4)

last_perc = -1
loss_list = list()
for epoch in range(NUM_EPOCHS):
	print('Epoch ', epoch)
	for i, (mask, features, label) in enumerate(loader):
		perc = (i*BATCH_SIZE*100)//len(descset)

		mask = cudalize(mask)
		features = cudalize(features)
		label = cudalize(label)

		logits = desc(mask, features)
		logp = logsm_fn(logits)
		loss = loss_fn(logp, label)

		opt.zero_grad()
		loss.backward()
		opt.step()

		if perc > last_perc:
			last_perc = perc
			loss_list.append(loss.item())
			print('{: 3d}% - {}'.format(perc, loss_list[-1]))

	if args.save:
		torch.save(desc.state_dict(), PT_NEW)
		print('Module saved')

print(loss_list)
with open(LOG_FILENAME,'w') as fd:
	json.dump(loss_list, fd)
