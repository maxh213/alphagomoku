import tensorflow as tf
import math
from training_data import process_training_data, get_files, get_test_files

#TODO: this should be got from the board file
# Width and Height of the board
BOARD_SIZE = 20

LEARNING_RATE = 0.01

TRAINING_DATA_FILE_COUNT = 2000
TEST_DATA_FILE_COUNT = 100

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

def get_weight_variable(shape):
	return tf.Variable(tf.random_uniform(shape, -1, 1))
	#initial = tf.truncated_normal(shape, stddev=0.1)
	#return tf.Variable(initial)

def get_bias_variable(shape):
	return tf.Variable(tf.random_uniform(shape, -0.1, 0.1))
	#initial = tf.constant(0.1, shape=shape)
	#return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 4, 1], padding='SAME')

def max_pool_2x2(x):
	#A 4-D Tensor with shape [batch, height, width, channels]
	#ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
	#strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
  return tf.nn.max_pool(x, ksize=[1, 2, 4, 1], 
  		strides=[1, 2, 4, 1], padding='SAME')  

def sigmoid(x):
	result = 1 / (1 + math.exp(-x))
	if result > 0.99999 :
		return 1
	elif result < 0.00001:
		return 0
	else:
		return result

def conv_network():
	print("Convolutional Neural Network training beginning...")
	print("If you see any 'can't read file' error messages below, please ignore them for now, this is normal behaviour in this early build")
	print("------------------------------------------------")

	# Get training data
	training_data = get_training_data()
	testing_data = get_test_data()

	# Set up placeholders for input and output
	training_input = tf.placeholder(tf.float32, [BOARD_SIZE, BOARD_SIZE])
	training_output = tf.placeholder(tf.float32, [BOARD_SIZE, BOARD_SIZE])

	# Initialise weights and biases
	#first layer
	#looking at a 5x5 grid at each point in the board_size
	#1 is the number of input channels so this shouldn't change
	conv_weights1 = get_weight_variable([5, 5, 1, 20])
	#bias is always the same as the last in the shape above
	conv_bias1 = get_bias_variable([20])

	#-1 and 1 are meant to be the colour channels of the image so no need to change them
	#20 by 20 is the boardsize
	input_image = tf.reshape(training_input, [-1,20,20,1])


	convolution1 = tf.nn.relu(conv2d(input_image, conv_weights1) + conv_bias1)
	pool1 = max_pool_2x2(convolution1)

	#second layer
	#here we'll have 40 features for each 5x5 patch and 20 input channels (one for each place on the board)
	#not 100% on this
	conv_weights2 = get_weight_variable([5, 5, 20, 200])
	conv_bias2 = get_bias_variable([200])

	convolution2 = tf.nn.relu(conv2d(pool1, conv_weights2) + conv_bias2)
	pool2 = max_pool_2x2(convolution2)

	fully_connected_weights1 = get_weight_variable([20, 1000])
	fully_connected_bias1 = get_bias_variable([1000])

	pool2_flat = tf.reshape(pool2, [-1, 20])
	fully_connected_output1 = tf.nn.relu(tf.matmul(pool2_flat, fully_connected_weights1) + fully_connected_bias1)

	#get rid of some neurons
	keep_prob = tf.placeholder(tf.float32)
	fully_connected1_drop = tf.nn.dropout(fully_connected_output1, keep_prob)

	fully_connected_weights2 = get_weight_variable([1000, 20])
	fully_connected_bias2 = get_bias_variable([20])

	output = tf.nn.softmax(tf.matmul(fully_connected1_drop, fully_connected_weights2) + fully_connected_bias2, 0)
	#output = tf.sigmoid(output)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, training_output))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(training_output,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	print_counter = 0
	# For each game
	for i in range (len(training_data)):
		# For each move in the game
		for j in range (len(training_data[i])):
			batch_input = training_data[i][j][0]
			batch_output = transform_training_output_for_tf(training_data[i][j][1])
			sess.run(train_step, feed_dict={training_input: batch_input, training_output: batch_output, keep_prob: 0.5})
			print_counter = print_counter + 1
			if print_counter % 1000 == 0:
				print("Output from network for likelyhood of -1 winning:")
				printable_output = sess.run(output, feed_dict={training_input: batch_input, training_output: batch_output, keep_prob: 1})
				print(printable_output[0][0])
				print("Output from network for likelyhood of 1 winning:")
				print(printable_output[0][1])
				print("Output from training data:")
				print(training_data[i][j][1])
				print("********************************")

	#TODO: move this to its own method
	#start accuracy calculation
	size = count_moves(testing_data)
	correct = 0
	same_guess_count = 0;
	for i in range(0, len(testing_data)):
		for j in range(0, len(testing_data[i])):
			test_input = testing_data[i][j][0]
			test_output = transform_training_output_for_tf(testing_data[i][j][1])
			comparable_output = sess.run(output, feed_dict={training_input: test_input, training_output: test_output, keep_prob: 1})
			print("---")
			print("chance of 1")
			print(comparable_output[0])
			print("chance of -1")
			print(comparable_output[0][0])
			print("answer")
			print(testing_data[i][j][1])
			if comparable_output[0][1] < comparable_output[0][0]:
				comparable_output = -1
			elif comparable_output[0][1] > comparable_output[0][0]:
				comparable_output = 1
			else:
				comparable_output = 0
				same_guess_count += 1

			if comparable_output == testing_data[i][j][1]:
				correct += 1

	print("same guesses made: %s" % same_guess_count)
	print("correct: %s" % (correct))
	print("number: %s" % size)
	accuracy = (correct / size) * 100
	print("%s percent" % (accuracy))
	#accuracy_without_same_guesses = (correct / (size - same_guess_count)) * 100
	#print("%s percent accuracy without same guesses" % (accuracy_without_same_guesses))

#This is now deprecated, we'll be editing the conv_network going forward
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
    weights = get_weight_variable([BOARD_SIZE, BOARD_SIZE])
    bias = get_bias_variable([BOARD_SIZE, BOARD_SIZE])

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
 

#Tensorflow cannot output a different shape than its input (which is a 20x20 board)
#so we have to do this so we can compare the actual training ouput with the tensorflow output
def transform_training_output_for_tf(actualTrainingOutput):
	output = []
	for i in range(BOARD_SIZE):
		output.append([])
		for j in range(BOARD_SIZE):
			if actualTrainingOutput == 1:
				if j % 1 == 0 and actualTrainingOutput == -1:
					output[i].append(10)
				elif j % 1 == 1 and actualTrainingOutput == 1:
					output[i].append(10)
				else:
					output[i].append(0)
	return output

if __name__ == '__main__':
    conv_network()