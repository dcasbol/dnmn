import os
import json
import torch
import subprocess
from misc.util import GPUScheduler
from glob import glob
from ilock import ILock

QUEUE_DIR = 'run-queue'
LOCK_NAME = 'run-next'

def pick_script():
	assert os.path.exists(QUEUE_DIR), "{} folder doesn't exist".format(QUEUE_DIR)
	with ILock(LOCK_NAME):
		scripts = sorted(glob(QUEUE_DIR+'/*.sh'))
		scripts = [ s for s in scripts if os.path.basename(s)[:4] != 'DONE' ]
		scripts = [ s for s in scripts if os.path.basename(s)[:7] != 'RUNNING' ]
		assert len(scripts) > 0, 'Run queue is empty'
		full_sname = scripts[0]
		sdir, sname = os.path.split(full_sname)
		new_sname = os.path.join(sdir, 'RUNNING-'+sname)
		os.rename(full_sname, new_sname)
		return full_sname, new_sname

def main():

	old_sname, sel_sname = pick_script()
	with GPUScheduler(old_sname) as gpu_id:
		sdir, sname = os.path.split(old_sname)

		print('Executing {!r}'.format(sname))
		with open(sel_sname) as fd:
			content = fd.read()
		print('\n'+content+'\n')

		env = os.environ.copy()
		if 'CUDA_VISIBLE_DEVICES' in env:
			del env['CUDA_VISIBLE_DEVICES']
		subprocess.run(['sh', sel_sname], env=env)
		print('\nExecution finished')

		done_sname = os.path.join(sdir, 'DONE-'+sname)
		os.rename(sel_sname, done_sname)

	print('GPU {} is now free'.format(gpu_id))

if __name__ == '__main__':
	main()
