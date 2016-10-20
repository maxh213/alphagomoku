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

def transformTrainingOutputForTF(actualTrainingOutput):
	return [
	    [actualTrainingOutput, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	]


def tensorMain():
	trainingData = getTrainingData()
	print(trainingData[0][0][0])
	move = tf.placeholder("float", [len(trainingData[0][1][0]),len(trainingData[0][1][0][0])])
	'''
	tensorflow only support comparing inputs and outputs of the same shapes!!
	So, we are passing it an output the size of the board but the first element in the board will be the actual output and the rest garbage
	By garbage I mean 0s
	'''
	gameWinner = tf.placeholder("float", [len(trainingData[0][1][0]),len(trainingData[0][1][0][0])])

	#the weights below mean there's a neuron for each place on the board
	firstLayerWeights = tf.Variable(tf.random_uniform([len(trainingData[0][1][0]),len(trainingData[0][1][0][0])], -0.1, 0.1))
	firstLayerBias = tf.Variable(tf.random_uniform([len(trainingData[0][1][0]),len(trainingData[0][1][0][0])], -0.1, 0.1))

	firstLayerOutput = tf.nn.relu(tf.matmul(move,firstLayerWeights) + firstLayerBias)

	#secondLayerWeights = tf.Variable(tf.random_uniform([len(trainingData[0][1][0]),len(trainingData[0][1][0][0])], -1, 1))
	#secondLayerOutput = tf.matmul(firstLayerOutput, secondLayerWeights)
	
	output = tf.nn.softmax(firstLayerOutput)
	crossEntropy = tf.reduce_mean(-tf.reduce_sum(gameWinner * tf.log(output), reduction_indices=[1]))
	#crossEntropy = -tf.reduce_sum(gameWinner*tf.log(output))

	trainStep = tf.train.GradientDescentOptimizer(0.2).minimize(crossEntropy)

	model = tf.initialize_all_variables()
	session = tf.Session()
	session.run(model)
	for i in range (len(trainingData)):
		for j in range (len(trainingData[i])):
			session.run(trainStep, feed_dict={move: trainingData[i][j][0], gameWinner:transformTrainingOutputForTF(trainingData[i][j][1])})
			entropy,_ = session.run([crossEntropy, trainStep], feed_dict={move: trainingData[i][j][0], gameWinner:transformTrainingOutputForTF(trainingData[i][j][1])})
			print("step ",i,",",j, ": entropy ", entropy)
			
			#The code below outputs the prediction made by the nn vs the actual training data output
			print("_------__----___----___--")
			#the output from tensorflow will always be between 0-1, we will have to cast it to between -1 and 1 later before it goes to the rest of the
			printableOuput = session.run(output, feed_dict={move: trainingData[i][j][0], gameWinner:transformTrainingOutputForTF(trainingData[i][j][1])})
			print(printableOuput[0][0])	
			print("_------__----___----___--")
			print(transformTrainingOutputForTF(trainingData[i][j][1])[0][0])
			print("_------__----___----___--")

	correctPrediction = tf.equal(tf.argmax(output,1), tf.argmax(gameWinner,1))
	accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float")) 
	print(session.run(accuracy, feed_dict={move: trainingData[0][40][0], gameWinner:transformTrainingOutputForTF(trainingData[0][40][1])}))

if __name__ == '__main__':
        tensorMain()



