import tensorflow as tf
import numpy as np
from training_data import process_training_data, get_files, get_test_files

#TODO: this should be got from the board file
# Width and Height of the board
BOARD_SIZE = 20

LEARNING_RATE = 0.1

TRAINING_DATA_FILE_COUNT = 2000
TEST_DATA_FILE_COUNT = 1000

# THIS BELONGS IN training_data.py
def count_moves(data):
    counter = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            counter += 1
    return counter

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
    return process_training_data(files[:TRAINING_DATA_FILE_COUNT])

def get_test_data():
    test_files = get_test_files()
    return process_training_data(test_files[:TEST_DATA_FILE_COUNT])

def network():
    print("Neural Network training beginning...")
    print("If you see any 'can't read file' error messages below, please ignore them for now, this is normal behaviour in this early build")
    print("------------------------------------------------")

    # Get training data
    training_data = get_training_data()
    testing_data = get_test_data()

    # Set up placeholders for input and output
    input = tf.placeholder(tf.float32, [BOARD_SIZE, BOARD_SIZE])
    correct = tf.placeholder(tf.float32, [BOARD_SIZE, BOARD_SIZE])

    # Initialise weights and biases
    weights = tf.Variable(tf.random_uniform([BOARD_SIZE, BOARD_SIZE], -0.1, 0.1))
    bias = tf.Variable(tf.random_uniform([BOARD_SIZE, BOARD_SIZE], -0.1, 0.1))

    # Define the model
    # the 0 argument at the end is the index you wish to apply softmax to
    # if you don't specify it assumes -1 which means the last entry
    model = tf.nn.softmax(tf.matmul(input, weights) + bias, 0)
    model = tf.sigmoid(model)

    # Define Cross Entropy function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct * tf.log(model)))

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
            sess.run(train_step, feed_dict={input: batch_input, correct: batch_output})
            print("********************************")
            print("Output from network:")
            printableOuput = sess.run(model, feed_dict={input: batch_input, correct: batch_output})
            print(printableOuput[0][0])
            print("Output from training data:")
            print(transform_training_output_for_tf(training_data[i][j][1])[0][0])
            print("********************************")
 
    #TODO: move this to its own method
    #start accuracy calculation
    size = count_moves(testing_data)
    correct = 0
    for i in range(0, len(testing_data)):
        for j in range(0, len(testing_data[i])):
            batch_input = testing_data[i][j][0]
            batch_output = transform_training_output_for_tf(testing_data[i][j][1])
            output = sess.run(model, feed_dict={input: batch_input})
            output = output[0][0]
            '''
                For some reason the sigmoid function only seems to go between 0.5 and 0.75
                It will literally never go above or below this

                but if we convert 0.5 to -1 and 0.75 to 1 
                so anything above 0.625 goes to 1 and anything below to -1
                it seems like it's actually learnt a bit
            '''
            if output < 0.625:
                output = -1
            else:
                output = 1
            if output == batch_output[0][0]:
                correct += 1

    print("correct: %s" % (correct))
    print("number: %s" % size)
    accuracy = (correct / size) * 100
    print("%s percent" % (accuracy))

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