import sys
import time
import glob
import argparse

def get_args():
	parser = argparse.ArgumentParser(
		description='Runs until a file is created or reaches time limit')
	parser.add_argument('filename')
	parser.add_argument('--check-interval', type=float, default=60.0)
	parser.add_argument('--time-limit', type=float, default=-1)
	return parser.parse_args()

def main(args):
	t0 = time.time()
	while len(glob.glob(args.filename)) == 0:
		time.sleep(args.check_interval)
		if args.time_limit > 0 and time.time()-t0 > args.time_limit:
			sys.exit(1)
	sys.exit(0)

if __name__ == '__main__':
	main(get_args())
