import tensorflow as tf
import numpy as np
from processTrainingData import processTrainingData
from getTrainingDataFiles import getFiles

#TODO: get this from the board file 
BOARD_SIZE = 20

'''
Training data format:
trainingData[0] = first game
trainingData[0][1][0] = first move of first game
trainingData[0][1][1] = winner of first game
trainingData[0][1][0][0] = first line of first move of first game
''' 
def getTrainingData():
	files = getFiles()
	files = files[:1] #for dev purposes just use the first however many
	return processTrainingData(files)

def tensorMain():
	trainingData = getTrainingData()
	#the weights below mean there's a neuron for each place on the board
	firstLayerWeights = tf.Variable(tf.random_uniform([len(trainingData[0][1][0]),len(trainingData[0][1][0][0])], -1, 1))
	
	move = tf.placeholder("float", [len(trainingData[0][1][0]),len(trainingData[0][1][0][0])])
	nextMove = tf.placeholder("float", [len(trainingData[0][1][0]),len(trainingData[0][1][0][0])])
	gameWinner = tf.placeholder("float", 1)

	model = tf.initialize_all_variables()
	with tf.Session() as session:
		tfMove = session.run(move, feed_dict={move: trainingData[0][1][0]})
		tfNextMove = session.run(nextMove, feed_dict={nextMove: trainingData[0][2][0]})
		tfGameWinner = session.run(gameWinner, feed_dict={gameWinner: [trainingData[0][1][1]]})
		print(tfMove)
		print("----")
		print(tfNextMove)
		print("----")
		print(tfGameWinner)

		session.run(model)
		print(session.run(firstLayerWeights))
		#Next step: Make a loop which will take in all the training data and learn

if __name__ == '__main__':
        tensorMain()



