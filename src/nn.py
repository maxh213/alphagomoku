import tensorflow as tf
import numpy as np
from processTrainingData import processTrainingData
from getTrainingDataFiles import getFiles

SIZE_OF_BOARD = 15

def tensorMain():
	files = getFiles()
	files = files[:20] #for dev purposes just use the first however many
	trainingData = processTrainingData(files)
	#print(trainingData)

	#Init NN variables
	##trainingGamePlaceholder = tf.placeholder("
	#this gets a list of random numbers between 0 - 1
	##randomStartingWeights = np.random.randint(2, size=SIZE_OF_BOARD)


	##LOGIC:
	#WHEN LEARN OFF THE NEXT MOVE MADE BY THE WINNING PLAYER

	#NEEDS:
	#THE GOMOKU APP TO FEED IT A LIST OF PLAYABLE MOVES!

if __name__ == '__main__':
        tensorMain()



