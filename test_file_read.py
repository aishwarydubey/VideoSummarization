import tensorflow as tf
import numpy as np
import os,glob,cv2,time
import sys,argparse
import operator
import cv2
fname='output/test.csv'
with open(fname) as f:
	content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
print(content)
frameNos=[]
scores=[]
for e in content:
    frameNoScore=e.split(",")
    score=float(frameNoScore[1])
    if score > 0.3:
        frameNos.append(int(frameNoScore[0]))
        scores.append(score)
print(frameNos)
print(scores)