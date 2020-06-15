import json
import argparse

def get_args():
	parser = argparse.ArgumentParser(description='Prints number of results in JSON')
	parser.add_argument('json')
	return parser.parse_args()

def main(args):

	with open(args.json) as fd:
		d = json.load(fd)

	for key, values in d.items():
		print(len(values))
		break

if __name__ == '__main__':
	main(get_args())
