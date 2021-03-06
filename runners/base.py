import torch
import misc.util as util
import numpy as np
from torch.utils.data import DataLoader
from vqa import VQAFindDataset, VQADescribeDataset, VQAMeasureDataset
from vqa import VQAEncoderDataset, encoder_collate_fn, nmn_collate_fn
from misc.constants import *
from misc.util import cudalize, USE_CUDA
from misc.util import Logger, Chronometer, PercentageCounter, time_iter
from model import GaugeFind


class Runner(object):

	def __init__(self, max_epochs=40, batch_size=128,
		restore_pt=None, save=False, validate=True, suffix='',
		learning_rate=1e-3, weight_decay=1e-5, dropout=0,
		early_stopping=True, modular=False, k=None):

		# This seed makes weight initialization deterministic
		self._seed()

		self._max_epochs = max_epochs
		self._save       = save or k is not None
		self._validate   = validate
		self._dropout    = dropout
		self._earl_stop  = early_stopping
		self._modular    = modular
		self._k          = k

		self._best_acc   = None
		self._test_acc   = None
		self._n_worse    = 0
		self._best_epoch = -1
		self._model      = self._get_model()

		modname = self._model.NAME
		suffix = '' if suffix == '' else '-' + suffix
		full_name = modname + suffix
		self._log_filename = full_name + '_log.json'
		self._pt_new       = full_name + '-new.pt'

		self._loader = self._get_loader(
			set_names   = 'train2014' if k is None else ['train2014','val2014'],
			partition   = None if k is None else 'train',
			batch_size  = batch_size,
			shuffle     = True,
			num_workers = 4
		)

		if validate or k is not None:
			if k is None:
				kwargs = {
					'set_names' : 'val2014',
					'stop'      : 0.2
				}
			else:
				kwargs = {
					'set_names' : ['train2014','val2014'],
					'k'         : k,
					'partition' : 'val'
				}
			self._val_loader = self._get_loader(
				batch_size  = VAL_BATCH_SIZE,
				shuffle     = False,
				num_workers = 4,
				**kwargs
			)

		self._logger   = Logger()
		keys = ['', 'raw_', 'stats_', 'val_', 'save_', 'batch_']
		self._clock    = { k+'time':Chronometer() for k in keys }
		self._perc_cnt = PercentageCounter(batch_size, self._loader.dataset_len)

		self._first_epoch = 0
		if restore_pt is not None:
			self._model.load(restore_pt)
			#self._logger.load(self._log_filename)
			#self._clock['time'].set_t0(self._logger.last('time'))
			#self._clock['raw_time'].set_t0(self._logger.last('raw_time'))
			#self._first_epoch = int(self._logger.last('epoch') + 0.5)

		self._opt = torch.optim.Adam(self._model.parameters(),
			lr=learning_rate, weight_decay=weight_decay)

	def _seed(self):
		torch.manual_seed(0)
		np.random.seed(0)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark     = False

	def _get_model(self):
		raise NotImplementedError

	def _loader_class(self):
		raise NotImplementedError

	def _forward(self, batch_data):
		raise NotImplementedError

	def run(self):
		# This seed ensures that training is deterministic
		self._seed()

		self._model = cudalize(self._model)
		self._logger.save(self._log_filename) # For HPO
		self._clock['raw_time'].start()
		for self._epoch in range(self._first_epoch, self._max_epochs):

			print('Epoch', self._epoch)
			N_perc = loss_perc = 0

			clk = self._clock['batch_time']
			for i, batch_data in time_iter(enumerate(self._loader), clk):

				self._clock['time'].start()
				result = self._forward(batch_data)
				output = result['output']

				loss = self._model.loss(output, result['label'])
				self._opt.zero_grad()
				loss.backward()
				self._opt.step()
				self._clock['time'].stop()

				self._clock['stats_time'].start()
				B = output.size(0)
				loss_perc += loss.item() * B
				N_perc    += B

				if loss_perc != loss_perc:
					print('Encountered nan. Training aborted.')
					return

				if self._perc_cnt.update(i):
					self._preview(loss_perc/N_perc)
				self._clock['stats_time'].stop()

			mean_loss = loss_perc/N_perc
			print('End of epoch', self._epoch)
			print('{} - {}'.format(self._clock['raw_time'].read_str(), mean_loss))

			self._validation_routine()
			self._log_routine(mean_loss)
			if self._evaluate(): break

		print('End of training. It took {} training seconds'.format(self._clock['time'].read()))
		print('{} seconds in total'.format(self._clock['raw_time'].read()))

		# Test here if k-folding
		self._test()

	def _preview(self, mean_loss):
		print('Ep. {}; {}; loss {}'.format(self._epoch, self._perc_cnt, mean_loss))

	def _log_routine(self, mean_loss):
		self._logger.log(
			epoch    = self._epoch,
			loss     = mean_loss
		)
		self._logger.log(**{ k:c.read() for k, c in self._clock.items() })
		self._logger.save(self._log_filename)

	def _validation_routine(self):
		if not self._validate: return
		self._clock['val_time'].start()
		print('Running validation...')
		is_gauge = self._model.NAME == GaugeFind.NAME
		N = top1 = relacc = var = loss = 0

		self._model.eval()
		with torch.no_grad():
			for batch_data in self._val_loader:
				result = self._forward(batch_data)
				output = result['output']
				label  = result['label']
				B = label.size(0)
				N += B
				top1 += util.top1_accuracy(output, label) * B
				loss += self._model.loss(output, label).item() * B
				relacc += util.rel_accuracy(output, label) * B
				if is_gauge:
					var += result['var'].sum().item()
		self._model.train()
		
		self._logger.log(top_1 = top1/N)
		self._logger.log(val_loss = loss/N)
		self._logger.log(rel_acc = relacc/N)
		if is_gauge:
			self._logger.log(var = var/N)
		print('...validation done')
		self._logger.print(exclude=['raw_time', 'time', 'epoch', 'loss'])
		self._clock['val_time'].stop()

	def _evaluate(self):
		if not self._validate:
			if self._save:
				self._clock['save_time'].start()
				self._model.save(self._pt_new)
				self._clock['save_time'].stop()
			return False

		acc = self._logger.last('top_1')
		if self._model.NAME == GaugeFind.NAME:
			var = self._logger.last('var')
			acc = acc if var < MAX_VARIANCE else 0.0

		if self._best_acc is None or acc > self._best_acc:
			self._best_acc = acc
			self._best_epoch = self._epoch
			self._n_worse = 0
			if self._save:
				self._clock['save_time'].start()
				self._model.save(self._pt_new)
				self._clock['save_time'].stop()
		else:
			self._n_worse += 1

		return self._earl_stop and self._n_worse >= max(5, (self._epoch+1)//3)

	def _test(self):
		if self._k is None: return
		if self._test_acc is not None:
			return self._test_acc

		self._model.load(self._pt_new)

		test_loader = self._get_loader(
			set_names   = ['train2014','val2014'],
			k           = self._k,
			partition   = 'test',
			batch_size  = VAL_BATCH_SIZE,
			shuffle     = False,
			num_workers = 4
		)

		N = top1 = 0

		self._model.eval()
		with torch.no_grad():
			for batch_data in test_loader:
				result = self._forward(batch_data)
				output = result['output']
				label  = result['label']
				B = label.size(0)
				N += B
				top1 += util.top1_accuracy(output, label) * B
		self._model.train()

		self._test_acc = top1/N
		return self._test_acc

	def save_model(self, filename):
		self._model.save(filename)

	@property
	def last_acc(self):
		return self._logger.last('top_1')

	@property
	def best_acc(self):
		if self._best_acc is None:
			print('WARNING: best_acc is not set')
			return 0.0
		return self._best_acc*100

	@property
	def test_acc(self):
		if self._test_acc is None:
			print('WARNING: test_acc is not set')
			return 0.0
		return self._test_acc*100
	
	@property
	def best_epoch(self):
		return self._best_epoch
	
	@property
	def log_filename(self):
		return self._log_filename

	@property
	def pt_filename(self):
		return self._pt_new
