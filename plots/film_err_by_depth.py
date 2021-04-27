import argparse
import json
import matplotlib.pyplot as plt

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('film_errors_json')
	parser.add_argument('--modular-errors-json')
	parser.add_argument('--classic-errors-json')
	return parser.parse_args()

def main(args):

	fig, ax1 = plt.subplots()
	insert_legend = False

	if args.modular_errors_json is not None:
		depths, errors = get_modular_errors(args.modular_errors_json, 'modular')
		ax1.plot(depths, errors, marker='^', color='#ff7f0e', label='Modular network with data types')
		insert_legend = True

	if args.classic_errors_json is not None:
		depths, errors = get_modular_errors(args.classic_errors_json, 'classic')
		ax1.plot(depths, errors, marker='o', color='#1f77b4', label='Modular baseline')
		insert_legend = True

	depths, errors = get_film_errors(args.film_errors_json)
	ax1.plot(depths, errors, marker='s', color='#2ca02c', label='FiLM')

	# Line marking max depth seen during train
	ax1.plot([5.5,5.5], [min(errors), max(errors)], linestyle='dashed', color='black')
	plt.xticks(list(range(min(depths), 18+1)))

	ax1.set_xlabel('Program depth')
	ax1.set_ylabel('Mean error (%)')
	if insert_legend:
		ax1.legend()
	plt.show()

def read_json(path):
	with open(path) as fd:
		return json.load(fd)

def get_film_errors(path):
	data = read_json(path)
	depths = [ int(k) for k in data.keys() ]
	depths.sort()
	errors = [ 100*(1-data[str(d)]) for d in depths if d <= 18 ]
	return depths, errors

def get_modular_errors(path, modality):
	with open(path) as fd:
		values = json.load(fd)[modality]
	depth, error_data = zip(*values.items())
	error, N = zip(*error_data)
	depth = [ int(d) for d in depth ]
	error = [ 100*e for e in error ]
	error = [ e for e,d in zip(error, depth) if d >= 3 and d <= 18 ]
	depth = [ d for d in depth if d >= 3 and d <= 18 ]
	return depth, error

if __name__ == '__main__':
	main(get_args())