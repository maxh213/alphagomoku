import tensorflow as tf
import numpy as np
from processTrainingData import processTrainingData
from getTrainingDataFiles import getFiles

def tensorMain():
	file1, file2, *rest = getFiles()
	print(file1)
	trainingData = processTrainingData(rest)
	print(trainingData)
	#print(trainingData) 
	#print(processTrainingData(["../resources/training/freestyle/freestyle1/0x2-28(1).psq"]))

if __name__ == '__main__':
        tensorMain()



