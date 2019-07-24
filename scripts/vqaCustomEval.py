# coding: utf-8

""" This script has been derived from the demo code provided at the official
VQA github, and adapted to work seemlessly with our implementation. For the
original code, see:
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
parser.add_argument('--data-dir', type=str, default='../data/vqa')
parser.add_argument('--result-dir', type=str, help='../results')
args = parser.parse_args()

vqa_dir = os.path.join(os.path.expanduser(args.vqa_dir), 'PythonHelperTools/vqaTools')

import sys
sys.path.insert(0, vqa_dir)
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

dataDir = os.path.expanduser(args.data_dir)
annFile     ='%s/Annotations/mscoco_%s_annotations.json'%(dataDir, args.set_name)
quesFile    ='%s/Questions/OpenEnded_mscoco_%s_questions.json'%(dataDir, args.set_name)
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 
 
res_dir = os.path.expanduser(args.result_dir)
res_dir += '/OpenEnded_mscoco_' + args.set_name + '_%s.json'
accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile = [ res_dir % ft for ft in fileTypes ]

vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(args.result_file, quesFile)
vqaEval = VQAEval(vqa, vqaRes, n=2) # n -> decimal places
vqaEval.evaluate() 

print("\n")
print("Overall Accuracy is: %.02f\n" % vqaEval.accuracy['overall'])
print("Per Question Type Accuracy is the following:")
for quesType in vqaEval.accuracy['perQuestionType']:
	print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
print("\n")
print("Per Answer Type Accuracy is the following:")
for ansType in vqaEval.accuracy['perAnswerType']:
	print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
print("\n")

plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation='0',fontsize=10)
plt.title('Per Question Type Accuracy', fontsize=10)
plt.xlabel('Question Types', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.show()

json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))

