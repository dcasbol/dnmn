# coding: utf-8

""" This script has been derived from the demo code provided at the official
VQA github, and adapted to work seemlessly with our implementation.

It must be called with Python 2 for compatibility reasons.

For the original code, see:
https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvalDemo.py
"""

import os
import json
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='VQA evaluation script')
parser.add_argument('set_name', type=str, choices=['train2014', 'val2014'])
parser.add_argument('result_file', type=str)
parser.add_argument('--vqa-dir', type=str, default='~/VQA')
parser.add_argument('--data-dir', type=str, default='~/DataSets/vqa')
parser.add_argument('--result-dir', type=str, default='/tmp')
parser.add_argument('--show-plot', action='store_true')
args = parser.parse_args()

vqa_help_dir = os.path.join(os.path.expanduser(args.vqa_dir), 'PythonHelperTools/vqaTools')
vqa_eval_dir = os.path.join(os.path.expanduser(args.vqa_dir), 'PythonEvaluationTools')

import sys
sys.path.insert(0, vqa_help_dir)
sys.path.insert(1, vqa_eval_dir)
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

dataDir = os.path.expanduser(args.data_dir)
annFile     ='%s/Annotations/mscoco_%s_annotations.json'%(dataDir, args.set_name)
quesFile    ='%s/Questions/OpenEnded_mscoco_%s_questions.json'%(dataDir, args.set_name)
fileTypes   = ['accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

res_dir = os.path.expanduser(args.result_dir)
res_dir += '/OpenEnded_mscoco_' + args.set_name + '_%s.json'
accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile = [ res_dir % ft for ft in fileTypes ]

vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(args.result_file, quesFile)
vqaEval = VQAEval(vqa, vqaRes, n=2) # n -> decimal places
quesIds = None
if args.set_name == 'val2014':
	quesIds = [ quesId for quesId in vqaEval.params['question_id'] ]
	quesIds.sort()
	quesIds = quesIds[int(0.2*len(quesIds)):]
vqaEval.evaluate(quesIds) 

print "\n"
print "Overall Accuracy is: %.02f\n" % vqaEval.accuracy['overall']
print "Per Question Type Accuracy is the following:" 
for quesType in vqaEval.accuracy['perQuestionType']:
	print "%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType])
print "\n"
print "Per Answer Type Accuracy is the following:"
for ansType in vqaEval.accuracy['perAnswerType']:
	print "%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType])
print "\n"

if args.show_plot:
	plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
	plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation='vertical',fontsize=10)
	plt.title('Per Question Type Accuracy', fontsize=10)
	plt.xlabel('Question Types', fontsize=10)
	plt.ylabel('Accuracy', fontsize=10)
	plt.show()

json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))

