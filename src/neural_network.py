import tensorflow as tf
import numpy as np
from training_data import process_training_data, get_files

#TODO: get this from the board file 
BOARD_SIZE = 20

'''
Training data format:
trainingData[0] = first game
trainingData[0][1][0] = first move of first game
trainingData[0][2][0] = second move of first game
trainingData[0][0][1] = winner of first game
trainingData[0][1][0][0] = first line of first move of first game
trainingData[0][1][0][0][0] = first tile on first line of first move of first game
''' 
def getTrainingData():
	files = get_files()
	files = files[:1] #for dev purposes just use the first however many
	return process_training_data(files)

#TODO: this should be in the board class but I can't import it, I tried my best, please someone else do it
#game = trainingData[i]
def getNextMoveOfWinningPlayer(game, currentMoveIndex):
	#Although this is a lot of loop, it will make at max 2 iterations through all of it
	for i in range (currentMoveIndex, len(game)):
		for j in range (0, len(game[i][0])):
			for k in range (0, len(game[i][0][j])):
				#If the current move and next move are different and the person who made the move is the winning player
				if ((game[i][0][j][k] != game[i+1][0][j][k]) and game[i][1] == game[i+1][0][j][k]):
					return [i+1, j, k]


def tensorMain():
	trainingData = getTrainingData()

	move = tf.placeholder("float", [len(trainingData[0][1][0]),len(trainingData[0][1][0][0])])
	gameWinner = tf.placeholder("float", 1)

	#the weights below mean there's a neuron for each place on the board
	firstLayerWeights = tf.Variable(tf.random_uniform([len(trainingData[0][1][0]),len(trainingData[0][1][0][0])], -1, 1))
	firstLayerBias = tf.Variable(tf.random_uniform([len(trainingData[0][1][0]),len(trainingData[0][1][0][0])], -0.1, 0.1))

	firstLayerOutput = tf.nn.relu(tf.matmul(move,firstLayerWeights) + firstLayerBias)

	secondLayerWeights = tf.Variable(tf.random_uniform([len(trainingData[0][1][0]),len(trainingData[0][1][0][0])], -1, 1))
	secondLayerOutput = tf.matmul(firstLayerOutput, secondLayerWeights)
	
	output = tf.nn.softmax(secondLayerOutput)
	#crossEntropy = -tf.reduce_sum(outputPlaceHolder*tf.log(output))
	#trainStep = tf.train.GradientDescentOptimizer(0.2).minimize(crossEntropy)

	

	model = tf.initialize_all_variables()
	with tf.Session() as session:
		tfMove = session.run(move, feed_dict={move: trainingData[0][1][0]})
		tfGameWinner = session.run(gameWinner, feed_dict={gameWinner: [trainingData[0][1][1]]})
		print(tfMove)
		print("----")
		print(tfGameWinner)

		session.run(model)
		weights = session.run(firstLayerWeights)
		print(weights)
		#Next step: Make a loop which will take in all the training data and learn
		print(getNextMoveOfWinningPlayer(trainingData[0], 5))

if __name__ == '__main__':
        tensorMain()



