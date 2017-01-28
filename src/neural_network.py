import tensorflow as tf
import math
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

LEARNING_RATE = 0.1
# The rate at which neurons are kept after learning
KEEP_PROBABILITY = 0.5

TRAINING_DATA_FILE_COUNT = 50
TEST_DATA_FILE_COUNT = 10


# THIS BELONGS IN training_data.py
def count_moves(data):
	counter = 0
	for i in range(len(data)):
		counter += len(data[i])
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
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def get_bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
	# [(Batch) 1, x, y, (Channels) 1]
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def avg_pool_2x2(x):
	# A 4-D Tensor with shape [batch, height, width, channels]
	# ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
	# strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_network():
	print("Convolutional Neural Network training beginning...")
	print("If you see any 'can't read file' error messages below, please ignore them for now, this is normal behaviour in this early build")
	print("------------------------------------------------")

	# Get training data
	training_data = get_training_data()
	testing_data = get_test_data()

	# Set up placeholders for input and output
	training_input = tf.placeholder(tf.float32, [BOARD_SIZE, BOARD_SIZE])
	training_output = tf.placeholder(tf.float32, [1, 100])

	# Initialise weights and biases
	# first layer
	# looking at a 5x5 grid at each point in the board_size
	# 1 is the number of input channels so this shouldn't change
	conv_weights1 = get_weight_variable([5, 5, 1, 32])
	# bias is always the same as the last in the shape above
	conv_bias1 = get_bias_variable([32])

	# -1 and 1 are meant to be the colour channels of the image so no need to change them
	# 20 by 20 is the boardsize
	input_image = tf.reshape(training_input, [-1, 20, 20, 1])

	convolution1 = tf.nn.relu(conv2d(input_image, conv_weights1) + conv_bias1)
	pool1 = avg_pool_2x2(convolution1)

	# second layer
	# here we'll have 40 features for each 5x5 patch and 20 input channels
	conv_weights2 = get_weight_variable([5, 5, 32, 64])
	conv_bias2 = get_bias_variable([64])

	convolution2 = tf.nn.relu(conv2d(pool1, conv_weights2) + conv_bias2)
	pool2 = avg_pool_2x2(convolution2)

	fully_connected_weights1 = get_weight_variable([1600, 1000])
	fully_connected_bias1 = get_bias_variable([1000])

	pool2_flat = tf.reshape(pool2, [-1, 1600])
	fully_connected_output1 = tf.nn.relu(tf.matmul(pool2_flat, fully_connected_weights1) + fully_connected_bias1)

	keep_prob = tf.placeholder(tf.float32)
	fully_connected1_drop = tf.nn.dropout(fully_connected_output1, keep_prob)

	fully_connected_weights2 = get_weight_variable([1000, 100])
	fully_connected_bias2 = get_bias_variable([100])

	tf_output = tf.nn.softmax(tf.matmul(fully_connected1_drop, fully_connected_weights2) + fully_connected_bias2, 0)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf_output, training_output))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(tf_output,1), tf.argmax(training_output,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	print("Network training starting!")
	print_counter = 0
	# For each game
	for i in range(len(training_data)):
		# For each move in the game
		for j in range(len(training_data[i])):
			batch_input = training_data[i][j][0]
			batch_output = transform_training_output_for_tf(training_data[i][j][1])
			entropy, _ = sess.run([cross_entropy,train_step], feed_dict={training_input: batch_input, training_output: batch_output, keep_prob: KEEP_PROBABILITY})
			if print_counter % 1000 == 0:
				print("Game Number: " + str(i))
				print("Entropy: " + str(entropy))
				output = sess.run(tf_output, feed_dict={training_input: batch_input, keep_prob: 1.0})
				correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(batch_output,1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
				print("Network output: " + str(sess.run(tf.argmax(output,1))))
				print("Actual output: " + str(sess.run(tf.argmax(batch_output,1))))
				print("Accuracy: " + str(sess.run(accuracy)))
				print("***")
			print_counter += 1

	print("NN training complete, moving on to testing.")

	# TODO: move this to its own method
	print_counter = 0
	size = count_moves(testing_data)
	correct = 0
	for i in range(0, len(testing_data)):
		for j in range(0, len(testing_data[i])):
			batch_input = testing_data[i][j][0]
			batch_output = transform_training_output_for_tf(testing_data[i][j][1])
			output = sess.run(tf_output, feed_dict={training_input: batch_input, keep_prob: 1.0})
			correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(batch_output,1))
			accuracy = sess.run(tf.reduce_mean(tf.cast(correct_prediction, "float")))
			if accuracy < 0.5:
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
	for i in range(1):
		output.append([])
		for j in range(100):
			if j == actual_training_output:
				output[i].append(100)
			else:
				output[i].append(0)
	return output


def onehot_input(input):
	input = []
	for row in input:
		for move in row:
			input.append(move)
	return input


if __name__ == '__main__':
	conv_network()
