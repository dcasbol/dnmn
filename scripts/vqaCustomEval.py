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
parser.add_argument('--vqa-dir', type=str, default='~/VQA')
parser.add_argument('--data-dir', type=str, default='../data/vqa')
args = parser.parse_args()

vqa_dir = os.path.join(os.path.expanduser(args.vqa_dir), 'PythonHelperTools/vqaTools')

import sys
sys.path.insert(0, vqa_dir)
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

dataDir = os.path.expanduser(args.data_dir)
annFile     ='%s/Annotations/mscoco_%s_annotations.json'%(dataDir, args.set_name)
quesFile    ='%s/Questions/OpenEnded_mscoco_%s_questions.json'%(dataDir, args.set_name)
resultType  ='fake'
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

# An example result json file has been provided in './Results' folder.  

[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/Results/%s%s_%s_%s_%s_%s.json'%(dataDir, versionType, taskType, dataType, dataSubType, \
resultType, fileType) for fileType in fileTypes]  

# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
"""
If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the question ids in annotation file
"""
vqaEval.evaluate() 

# print accuracies
print "\n"
print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
print "Per Question Type Accuracy is the following:"
for quesType in vqaEval.accuracy['perQuestionType']:
	print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
print "\n"
print "Per Answer Type Accuracy is the following:"
for ansType in vqaEval.accuracy['perAnswerType']:
	print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
print "\n"

# plot accuracy for various question types
plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation='0',fontsize=10)
plt.title('Per Question Type Accuracy', fontsize=10)
plt.xlabel('Question Types', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.show()

# save evaluation results to ./Results folder
json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))

