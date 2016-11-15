import tensorflow as tf
import math
import random
from training_data import process_training_data, get_files, get_test_files

'''
-make the numbers make sense mathematically, and make it compile with said numbers
-figure out trunticated normal should we use it???
-figure out what our output data should look like when we feed to to tensorflow.
- Do we need 0 padding for the windows?
- Which version of pooling do we need, seeing as max pooling doesn't sit with our input data
	- It looks like average pooling might help!
	- Actually no, because this is just going to return the player with the most moves in a patch
'''

# TODO: this should be got from the board file
# Width and Height of the board
BOARD_SIZE = 20

LEARNING_RATE = 1e-6
# The rate at which neurons are kept after learning
KEEP_PROBABILITY = 0.5

TRAINING_DATA_FILE_COUNT = 1000
TEST_DATA_FILE_COUNT = 20


# THIS BELONGS IN training_data.py
def count_moves(data):
	counter = 0
	for i in range(len(data)):
		counter += len(data[i])
	return counter


'''
Training data format:
training_data[0] = first game
training_data[0][1][0] = first move of first game
training_data[0][2][0] = second move of first game
training_data[0][0][1] = winner of first game
training_data[0][1][0][0] = first line of first move of first game
training_data[0][1][0][0][0] = first tile on first line of first move of first game

After the training data has been shuffled training data loses the first index above and they're just random moves from any given game
For example:
training_data[0][0] = a move of a random game
training_data[0][1] = winner of the game which the random move was taken from
training_data[1][0] = another move of a random game
and so on...
'''
def get_training_data(file_count):
	# Obtain files for processing
	files = get_files()
	return process_training_data(files[:file_count])


def get_test_data(file_count):
	test_files = get_test_files()
	return process_training_data(test_files[:file_count])


def get_weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def get_bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
	# [(Batch) 1, x, y, (Channels) 1]
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


#def max_pool_2x2(x):
	# A 4-D Tensor with shape [batch, height, width, channels]
	# ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
	# strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
	#return tf.nn.avg_pool(x, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')


def conv_network():
	print("Convolutional Neural Network training beginning...")
	print("If you see any 'can't read file' error messages below, please ignore them for now, this is normal behaviour in this early build")
	print("------------------------------------------------")

	# Get training data
	training_data = get_training_data(TRAINING_DATA_FILE_COUNT)
	training_data = shuffle_training_data(training_data)
	testing_data = get_test_data(TEST_DATA_FILE_COUNT)

	# Set up placeholders for input and output
	training_input = tf.placeholder(tf.float32, [BOARD_SIZE, BOARD_SIZE])
	training_output = tf.placeholder(tf.float32, [400, 2])

	# Initialise weights and biases
	# first layer
	# looking at a 5x5 grid at each point in the board_size
	# 1 is the number of input channels so this shouldn't change
	conv_weights1 = get_weight_variable([5, 5, 1, 5])
	# bias is always the same as the last in the shape above
	conv_bias1 = get_bias_variable([5])

	# -1 and 1 are meant to be the colour channels of the image so no need to change them
	# 20 by 20 is the boardsize
	input_image = tf.reshape(training_input, [-1, 20, 20, 1])

	convolution1 = tf.nn.relu(conv2d(input_image, conv_weights1) + conv_bias1)
	#pool1 = max_pool_2x2(convolution1)

	# second layer
	conv_weights2 = get_weight_variable([5, 5, 5, 10])
	conv_bias2 = get_bias_variable([10])

	convolution2 = tf.nn.relu(conv2d(convolution1, conv_weights2) + conv_bias2)
	#pool2 = max_pool_2x2(convolution2)

	fully_connected_weights1 = get_weight_variable([10, 500])
	fully_connected_bias1 = get_bias_variable([500])

	#pool2_flat = tf.reshape(pool2, [-1, 10])
	conv2_flat = tf.reshape(convolution2, [-1, 10])
	fully_connected_output1 = tf.nn.relu(tf.matmul(conv2_flat, fully_connected_weights1) + fully_connected_bias1)

	keep_prob = tf.placeholder(tf.float32)
	fully_connected1_drop = tf.nn.dropout(fully_connected_output1, keep_prob)

	fully_connected_weights2 = get_weight_variable([500, 2])
	fully_connected_bias2 = get_bias_variable([2])

	tf_output = tf.nn.softmax(tf.matmul(fully_connected1_drop, fully_connected_weights2) + fully_connected_bias2)
	tf_output = tf.sigmoid(tf_output)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf_output, training_output))
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(tf_output,1), tf.argmax(training_output,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	print("Network training starting!")
	print_counter = 0
	# For each game

	for i in range(0, len(training_data)):
		batch_input = training_data[i][0]
		batch_output = transform_training_output_for_tf(training_data[i][1])
		entropy, _ = sess.run([cross_entropy,train_step], feed_dict={training_input: batch_input, training_output: batch_output, keep_prob: KEEP_PROBABILITY})
		if print_counter % 100 == 0:
			print("Training Data Number: " + str(i) + "/" + str(len(training_data)))
			print("Entropy: " + str(entropy))
			output = sess.run(tf_output, feed_dict={training_input: batch_input, keep_prob: 1.0})
			correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(batch_output,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			#print("Network output: " + str(sess.run(tf.argmax(output,1))))
			#print("Actual output: " + str(sess.run(tf.argmax(batch_output,1))))
			print("Accuracy: " + str(sess.run(accuracy)))
			print("***")
		print_counter += 1

	print("NN training complete, moving on to testing.")

	# TODO: move this to its own method
	print_counter = 0
	size = count_moves(testing_data)
	correct = 0
	for i in range(0, len(testing_data)):
		if i % 2 == 0:
			print("Tested " + str(i) + "/" + str(len(testing_data)) + " games...")
		starting_move = 0
		if (len(testing_data[i]) > 10):
			starting_move = 10
		for j in range(starting_move, len(testing_data[i])):
			batch_input = testing_data[i][j][0]
			batch_output = transform_training_output_for_tf(testing_data[i][j][1])
			output = sess.run(tf_output, feed_dict={training_input: batch_input, keep_prob: 1.0})
			correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(batch_output,1))
			accuracy = sess.run(tf.reduce_mean(tf.cast(correct_prediction, "float")))
			if accuracy >= 0.5:
				correct += 1


	print("correct: %s" % (correct))
	print("number: %s" % size)
	percentage = (correct / size) * 100
	print("%s percent" % (percentage))


def transform_training_output_for_tf(actual_training_output):
	if actual_training_output == -1:
		actual_training_output = 0
	output = []
	count = 0
	for i in range(400):
		output.append([])
		for j in range(2):
			if j == actual_training_output:
				output[i].append(100)
			else:
				output[i].append(0)
	return output

'''
	This shuffles all the moves from all the games in the training data into one array.
	This is done so we don't bombard our NN with hundreds of the same training outputs at once.
'''
def shuffle_training_data(training_data):
	new_training_data = []
	for i in range(len(training_data)):

		# Here we remove the first 10 starting moves (for now!)
		# not much can be gained from the first 10 or so moves (I think!),
		# and it messes with the weights/bias unnecessarily.
		starting_move = 0
		if len(training_data[i]) > 10:
			starting_move = 10

		for j in range(starting_move, len(training_data[i])):
			new_training_data.append(training_data[i][j])
	random.shuffle(new_training_data)
	return new_training_data

if __name__ == '__main__':
	conv_network()
