import tensorflow as tf
import numpy as np
from training_data import process_training_data, get_files, get_test_files

#TODO: this should be got from the board file
# Width and Height of the board
BOARD_SIZE = 20

LEARNING_RATE = 0.1

FILE_COUNT = 100

'''
Training data format:
trainingData[0] = first game
trainingData[0][1][0] = first move of first game
trainingData[0][2][0] = second move of first game
trainingData[0][0][1] = winner of first game
trainingData[0][1][0][0] = first line of first move of first game
trainingData[0][1][0][0][0] = first tile on first line of first move of first game
''' 
def get_training_data():
    # Obtain files for processing
    files = get_files()
    return process_training_data(files[:FILE_COUNT])

def get_test_data():
    test_files = get_test_files()
    return process_training_data(test_files[:FILE_COUNT])

def network():
    # Get training data
    training_data = get_training_data()
    testing_data = get_test_data()

    # Set up placeholders for input and output
    input = tf.placeholder(tf.float32, [BOARD_SIZE, BOARD_SIZE])
    correct = tf.placeholder(tf.float32, [BOARD_SIZE, BOARD_SIZE])

    # Initialise weights and biases
    W = tf.Variable(tf.zeros([BOARD_SIZE, BOARD_SIZE]))
    b = tf.Variable(tf.zeros([BOARD_SIZE, BOARD_SIZE]))

    # Define the model
    model = tf.nn.softmax(tf.matmul(input, W) + b)

    # Define Cross Entropy function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct * tf.log(model), reduction_indices=[1]))

    # Define a training step
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    # Prepare model for training
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)


    # For each game
    for i in range (len(training_data)):
        # For each move in the game
        for j in range (len(training_data[i])):
            batch_input = training_data[i][j][0]
            batch_output = transform_training_output_for_tf(training_data[i][j][1])
            #print("Game: %s, Move: %s, Winner: %s" % (i + 1, j + 1, batch_output[0][0]))
            sess.run(train_step, feed_dict={input: batch_input, correct: batch_output})

            print("********************************")
            print("Ouput from network:")
            printableOuput = sess.run(model, feed_dict={input: batch_input, correct: batch_output})
            print(printableOuput[0][0]) 
            print("Output from training data:")
            print(transform_training_output_for_tf(training_data[i][j][1])[0][0])
            print("********************************")

    # Determine accuracy
    correct_predicition = tf.equal(tf.argmax(model, 1), tf.argmax(correct, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))
    
    accuracy_list = []
     # For each game
    for i in range (0, len(testing_data)):
        # For each move in the game
        for j in range (0, len(testing_data[i])):
            batch_input = testing_data[i][j][0]
            batch_output = transform_training_output_for_tf(testing_data[i][j][1])
            printableOuput = sess.run(model, feed_dict={input: batch_input, correct: batch_output})
            accuracy_list.append([testing_data[i][j][1], printableOuput[0][0], sess.run(accuracy, feed_dict={input: testing_data[i][j][0], correct: transform_training_output_for_tf(testing_data[i][j][1])})])
            
    #total_accuracy = sum(accuracy_list) / float(len(accuracy_list))
    print(accuracy_list)
    #print("Neural Network Accuracy: ", total_accuracy)


def transform_training_output_for_tf(actualTrainingOutput):
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



if __name__ == '__main__':
    network()