import torch
import misc.util as util
from torch.utils.data import DataLoader
from vqa import VQADescribeDataset, VQAMeasureDataset
from vqa import VQAEncoderDataset, encoder_collate_fn
from misc.constants import *
from misc.util import cudalize, Logger, Chronometer, PercentageCounter

class Runner(object):

	def __init__(self, max_epochs=100, batch_size=512, 
		restore=False, save=False, validate=False, suffix='',
		learning_rate=1e-3, weight_decay=1e-2):

		self._save      = save
		self._validate  = validate
		self._best_acc  = 0.
		self._n_worse   = 0
		self._best_epoch = -1

		modname = self._model.NAME
		suffix = '' if suffix == '' else '-' + suffix
		full_name    = modname + suffix
		self._log_filename = full_name + '_log.json'
		self._pt_restore   = full_name + '.pt'
		self._pt_new       = full_name + '-new.pt'

		kwargs = dict(
			batch_size  = batch_size,
			shuffle     = True,
			num_workers = 4
		)
		if modname == 'encoder':
			kwargs['collate_fn'] = encoder_collate_fn
		self._loader = DataLoader(self._dataset, **kwargs)

		if validate:
			valset = dict(
				describe = VQADescribeDataset,
				measure  = VQAMeasureDataset,
				encoder  = VQAEncoderDataset,
			)[modname](set_names='val2014', stop=0.1)
			kwargs = dict(collate_fn=encoder_collate_fn) if modname == 'encoder' else {}
			self._val_loader = DataLoader(valset,
				batch_size = VAL_BATCH_SIZE, shuffle = False, **kwargs)

		if modname == 'find':
			self._loss_fn = lambda a, b: self._model.loss()
		elif modname == 'nmn':
			def cross_entropy(x, y):
				# L(x) = -y*log(x) -(1-y)*log(1-x)
				x = x[torch.arange(x.size(0)), y]
				ce = -((x+1e-10).log() + (1.-x+1e-10).log()).sum()
				return ce
			self._loss_fn = cross_entropy
		else:
			self._loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

		self._logger    = Logger()
		self._clock     = Chronometer()
		self._raw_clock = Chronometer()
		self._perc_cnt  = PercentageCounter(batch_size, len(self._dataset))

		self._max_epochs = max_epochs
		self._first_epoch = 0
		if restore:
			self._logger.load(self._log_filename)
			self._model.load_state_dict(torch.load(self._pt_restore, map_location='cpu'))
			self._clock.set_t0(self._logger.last('time'))
			self._raw_clock.set_t0(self._logger.last('raw_time'))
			self._first_epoch = int(self._logger.last('epoch') + 0.5)

		self._model = cudalize(self._model)
		self._opt = torch.optim.Adam(self._model.parameters(),
			lr=learning_rate, weight_decay=weight_decay)

	def _forward(self, batch_data):
		raise NotImplementedError

	def run(self):
		self._raw_clock.start()
		for self._epoch in range(self._first_epoch, self._max_epochs):

			print('Epoch', self._epoch)
			loss_perc = 0.
			N_perc    = 0

			for i, batch_data in enumerate(self._loader):

				# ---   begin timed block   ---
				self._clock.start()
				result = self._forward(batch_data)
				output = result['output']

				loss = self._loss_fn(output, result['label'])
				self._opt.zero_grad()
				loss.backward()
				self._opt.step()
				self._clock.stop()
				# ---   end timed block   ---

				loss_perc += loss.item()
				N_perc    += output.size(0)

				if loss_perc != loss_perc:
					print('Encountered nan. Aborted')
					return

				if self._perc_cnt.update(i):
					mean_loss  = loss_perc/N_perc
					loss_perc  = 0.
					N_perc     = 0
					self._log_routine(mean_loss)
					self._validation_routine()

			if self._evaluate(): break

		print('End of training. It took {} training seconds'.format(self._clock.read()))
		print('{} seconds in total'.format(self._raw_clock.read()))	

	def _log_routine(self, mean_loss):
		self._logger.log(
			raw_time = self._raw_clock.read(),
			time     = self._clock.read(),
			epoch    = self._epoch + self._perc_cnt.float(),
			loss     = mean_loss
		)
		raw_tstr = self._raw_clock.read_str()
		tstr     = self._clock.read_str()
		print('{}/{} {} - {}'.format(raw_tstr, tstr, self._perc_cnt, mean_loss))

	def _validation_routine(self):
		if not self._validate: return
		N = top1 = inset = wacc = 0

		self._model.eval()
		for batch_data in self._val_loader:
			result = self._forward(batch_data)
			output = result['output'].softmax(1)
			label  = result['label']
			distr  = result['distr']
			B = label.size(0)
			N += B
			top1  += util.top1_accuracy(output, label) * B
			inset += util.inset_accuracy(output, distr) * B
			wacc  += util.weighted_accuracy(output, distr) * B
			break
		self._model.train()
		
		self._logger.log(
			top_1    = top1/N,
			in_set   = inset/N,
			weighted = wacc/N
		)
		self._logger.print(exclude=['raw_time', 'time', 'epoch'])

	def _evaluate(self):
		self._logger.save(self._log_filename)

		if not self._validate:
			if self._save:
				torch.save(self._model.state_dict(), self._pt_new)
				print('Model saved')
			return False

		acc = sum(self._logger._log['top_1'][-10:])/10
		if acc > self._best_acc:
			self._best_acc = acc
			self._best_epoch = self._epoch
			self._n_worse = 0
			if self._save:
				torch.save(self._model.state_dict(), self._pt_new)
				print('Model saved')
		else:
			self._n_worse += 1

		return self._n_worse >= 3

	@property
	def best_acc(self):
		return self._best_acc*100
	
	@property
	def best_epoch(self):
		return self._best_epoch
	