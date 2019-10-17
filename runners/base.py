import torch
import misc.util as util
from torch.utils.data import DataLoader
from vqa import VQAFindDataset, VQADescribeDataset, VQAMeasureDataset
from vqa import VQAEncoderDataset, encoder_collate_fn, nmn_collate_fn
from misc.constants import *
from misc.util import cudalize, Logger, Chronometer, PercentageCounter
from modules import GaugeFind


class Runner(object):

	def __init__(self, max_epochs=40, batch_size=128,
		restore=False, save=False, validate=True, suffix='',
		learning_rate=1e-3, weight_decay=1e-5, dropout=0):

		self._max_epochs = max_epochs
		self._save       = save
		self._validate   = validate
		self._dropout    = dropout

		self._best_acc   = None
		self._n_worse    = 0
		self._best_epoch = -1
		self._model      = self._get_model()

		modname = self._model.NAME
		suffix = '' if suffix == '' else '-' + suffix
		full_name = modname + suffix
		self._log_filename = full_name + '_log.json'
		self._pt_restore   = full_name + '.pt'
		self._pt_new       = full_name + '-new.pt'

		loader_class = self._loader_class()
		self._loader = loader_class(
			batch_size  = batch_size,
			shuffle     = True,
			num_workers = 4
		)

		if validate:
			#kwargs = dict(metadata=True) if modname == GaugeFind.NAME and validate else {}
			self._val_loader = loader_class(
				set_names  = 'val2014',
				stop       = 0.2,
				batch_size = VAL_BATCH_SIZE,
				shuffle    = False,
				#**kwargs
			)

		self._logger    = Logger()
		self._clock     = Chronometer()
		self._raw_clock = Chronometer()
		self._perc_cnt  = PercentageCounter(batch_size, self._loader.dataset_len)

		self._first_epoch = 0
		if restore:
			self._model.load(self._pt_restore)
			self._logger.load(self._log_filename)
			self._clock.set_t0(self._logger.last('time'))
			self._raw_clock.set_t0(self._logger.last('raw_time'))
			self._first_epoch = int(self._logger.last('epoch') + 0.5)

		self._model = cudalize(self._model)
		self._opt = torch.optim.Adam(self._model.parameters(),
			lr=learning_rate, weight_decay=weight_decay)

	def _get_model(self):
		raise NotImplementedError

	def _loader_class(self):
		raise NotImplementedError

	def _forward(self, batch_data):
		raise NotImplementedError

	def run(self):
		self._raw_clock.start()
		for self._epoch in range(self._first_epoch, self._max_epochs):

			print('Epoch', self._epoch)
			N_perc = loss_perc = 0

			for i, batch_data in enumerate(self._loader):

				# ---   begin timed block   ---
				self._clock.start()
				result = self._forward(batch_data)
				output = result['output']

				loss = self._model.loss(output, result['label'])
				self._opt.zero_grad()
				loss.backward()
				self._opt.step()
				self._clock.stop()
				# ---   end timed block   ---

				loss_perc += loss.item()
				N_perc    += output.size(0)

				if loss_perc != loss_perc:
					print('Encountered nan. Training aborted.')
					return

				if self._perc_cnt.update(i):
					self._preview(loss_perc/N_perc)

			mean_loss = loss_perc/N_perc
			self._log_routine(mean_loss)
			self._validation_routine()
			if self._evaluate(): break

		print('End of training. It took {} training seconds'.format(self._clock.read()))
		print('{} seconds in total'.format(self._raw_clock.read()))

	def _preview(self, mean_loss):
		print('Ep. {}; {}; loss {}'.format(self._epoch, self._perc_cnt, mean_loss))

	def _log_routine(self, mean_loss):
		self._logger.log(
			raw_time = self._raw_clock.read(),
			time     = self._clock.read(),
			epoch    = self._epoch,
			loss     = mean_loss
		)
		raw_tstr = self._raw_clock.read_str()
		tstr     = self._clock.read_str()
		print('End of epoch', self._epoch)
		print('{}/{} - {}'.format(raw_tstr, tstr, mean_loss))

	def _validation_routine(self):
		if not self._validate: return
		N = top1 = 0

		self._model.eval()
		with torch.no_grad():
			for batch_data in self._val_loader:
				result = self._forward(batch_data)
				output = result['output'].softmax(1)
				label  = result['label']
				B = label.size(0)
				N += B
				top1  += util.top1_accuracy(output, label) * B
		self._model.train()
		
		self._logger.log(top_1 = top1/N)
		self._logger.print(exclude=['raw_time', 'time', 'epoch', 'loss'])

	def _evaluate(self):
		self._logger.save(self._log_filename)

		if not self._validate:
			if self._save:
				self._model.save(self._pt_new)
			return False

		acc = self._logger.last('top_1')

		if self._best_acc is None or acc > self._best_acc:
			self._best_acc = acc
			self._best_epoch = self._epoch
			self._n_worse = 0
			if self._save:
				self._model.save(self._pt_new)
		else:
			self._n_worse += 1

		return self._n_worse >= max(3, (self._epoch+1)//3)

	def save_model(self, filename):
		self._model.save(filename)

	@property
	def best_acc(self):
		return self._best_acc*100
	
	@property
	def best_epoch(self):
		return self._best_epoch
	
