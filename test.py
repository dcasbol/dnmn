import argparse
import torch
import json
from vqatorch import VQADataset
from modules import FindModule, DescribeModule
from misc.util import cudalize
from misc.indices import ANSWER_INDEX, QUESTION_INDEX

if __name__ == '__main__':

	# ----------------------------------
	# PENDING INTEGRATION OF LSTM MODULE
	# Merge outputs with geom. average
	# ----------------------------------

	parser = argparse.ArgumentParser(description='Test a composed model')
	parser.add_argument('find_module', type=str)
	parser.add_argument('describe_module', type=str)
	parser.add_argument('--output', type=str, default='results.json')
	args = parser.parse_args()

	dataset = VQADataset('./', 'test2015')

	find = FindModule()
	desc = DescribeModule()
	find.eval()

	find.load_state_dict(torch.load(args.find_module, map_location='cpu'))
	desc.load_state_dict(torch.load(args.describe_module, map_location='cpu'))

	find = cudalize(find)
	desc = cudalize(desc)

	answer_list = list()

	#for i in range(len(dataset)):
	for i in range(100):
		datum, features = dataset[i]


		features = cudalize(torch.tensor(features)).unsqueeze(0)

		names = datum['layouts_names']
		indices = datum['layouts_indices']

		find_indices = [ idx for idx, name in zip(indices, names) if name == 'find' ]
		find_indices = cudalize(torch.tensor(find_indices)).view(-1)

		mask = find(features, find_indices).prod(1, keepdim=True)
		pred = desc(mask, features)

		answer = ANSWER_INDEX.get(pred.argmax().item())
		answer_list.append(dict(question_id=datum['question_id'], answer=answer))

		readable = ' '.join([ QUESTION_INDEX.get(idx) for idx in datum['question'] ])
		print(readable+'?', answer)

	print('Writing results to {!r}'.format(args.output))
	with open(args.output, 'w') as fd:
		json.dump(answer_list, fd)
	print('Finished')

"""
Evaluate using code here in
https://github.com/GT-Vision-Lab/VQA
https://github.com/VT-vision-lab/VQA/blob/master/PythonEvaluationTools/vqaEvalDemo.py
"""