import numpy
import tensorflow as tf
import math
import random
from random import randrange
from training_data import get_training_data, get_test_data
from board import BOARD_SIZE
from sys import argv

#---FILE BASED CONSTANTS---
DEBUG_PRINT_SIZE = 5
'''
	It's very possible the program will crash if you decrease NUMBER_OF_BATCHES due to an out of memory.
	This is because NUMBER_OF_BATCHES is how many times the total training/testing data is split into separate batches,
	so the lower the number the larger the batch amount.

	Decrease NUMBER_OF_BATCHES_TO_TRAIN_ON if you don't wish to train on every batch. 
	NUMBER_OF_BATCHES_TO_TRAIN_ON should be no larger than NUMBER_OF_BATCHES
'''
NUMBER_OF_BATCHES = 2000
NUMBER_OF_BATCHES_TO_TRAIN_ON = 5
#This is how many times each batch will be trained on
TRAINING_ITERATIONS = 5

MODEL_SAVE_FILE_PATH = "save_data/models/model.ckpt"
GRAPH_LOGS_SAVE_FILE_PATH = "save_data/logs/"

#---HYPER PARAMETERS ---
LEARNING_RATE = 1e-4
#below is only needed for gradient decent
#DECAY_LEARNING_RATE_EVERY_N = math.ceil(TRAINING_ITERATIONS/4)
#DECAY_RATE = 0.96

# The rate at which neurons are kept after learning
KEEP_SOME_PROBABILITY = 0.7
KEEP_ALL_PROBABILITY = 1.0

#Setting the below to None means load all of them
TRAINING_DATA_FILE_COUNT = None 
TEST_DATA_FILE_COUNT = None

#--- LAYER/WEIGHT/BIAS CONSTANTS---
INPUT_SIZE = BOARD_SIZE ** 2
OUTPUT_SIZE = 2
CONV_SIZE = 5
CONV_WEIGHT_1_INPUT_CHANNELS = 1
CONV_WEIGHT_1_FEATURES = 30
CONV_WEIGHT_2_FEATURES = 30
#The board size in the below conv outbout size is usually divided by 4 because of 2x2 pooling but we don't do that which means we have a huge amount of neurons
CONV_2_OUTPUT_SIZE = BOARD_SIZE * BOARD_SIZE * CONV_WEIGHT_2_FEATURES 
FC_LAYER_1_WEIGHTS = 10000
STRIDE_SIZE = 1 #this probably won't need changing
COLOUR_CHANNELS_USED = 1 #We are not feeding our network a colour image so this is always 1


def get_weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def get_bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

'''
	This only works on batched_inputs
'''
def one_hot_input_batch(input_batch):
	one_hotted_input_batch = []
	for board in input_batch:
		one_hotted_move = []
		for row in board:
			for cell in row:
				one_hotted_move.append(cell)
		one_hotted_input_batch.append(one_hotted_move)
	return one_hotted_input_batch

'''
	returns the training data in a batch format which can be argmaxed by tensorflow
'''
def convert_training_to_batch(training_data, number_of_batches):
	train_input = []
	train_output = []
	for i in range(len(training_data)):
		for j in range(len(training_data[i])):
			# if the move number is less than 5 and the game lasts more than 5 moves don't bother
			if not (j < 5 and len(training_data[i]) > 5):
				train_input.append(training_data[i][j][0])
				# If training_data[i][j][1] == -1 then an argmax function would identify the first index 0 as the highest
				# If training_data[i][j][1] == 1 then the argmax function would identify index 1 as the highest
				# Our nn just has to mimic this
				train_output.append([0, training_data[i][j][1]])
	train_input, train_output = shuffle(train_input, train_output)
	if number_of_batches == 1:
		return train_input, train_output
	else:
		return split_list_into_n_lists(train_input, number_of_batches), split_list_into_n_lists(train_output, number_of_batches)

def split_list_into_n_lists(list, n):
	return [list[i::n] for i in range(n)]


def conv2d(image, weights):
	return tf.nn.conv2d(image, weights, strides=[STRIDE_SIZE, STRIDE_SIZE, STRIDE_SIZE, STRIDE_SIZE], padding='SAME')

def neural_network_train(should_use_save_data):
	print("Convolutional Neural Network training beginning...")

	print("Loading training data...")
	training_data = get_training_data(TRAINING_DATA_FILE_COUNT)
	testing_data = get_test_data(TEST_DATA_FILE_COUNT)
	print("Training data loaded!")

	training_input = tf.placeholder(tf.float32, [None, INPUT_SIZE])
	training_output = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
	keep_prob = tf.placeholder(tf.float32)
	global_step = tf.Variable(0, trainable=False)

	conv_weights1 = get_weight_variable([CONV_SIZE, CONV_SIZE, CONV_WEIGHT_1_INPUT_CHANNELS, CONV_WEIGHT_1_FEATURES])
	conv_bias1 = get_bias_variable([CONV_WEIGHT_1_FEATURES])

	layer_1_weights_histogram = tf.histogram_summary("conv_weights1", conv_weights1)
	layer_1_bias_histogram = tf.histogram_summary("conv_bias1", conv_bias1)

	input_image = tf.reshape(training_input, [-1, BOARD_SIZE, BOARD_SIZE, COLOUR_CHANNELS_USED])

	convolution1 = tf.nn.tanh(conv2d(input_image, conv_weights1) + conv_bias1)

	conv_weights2 = get_weight_variable([CONV_SIZE, CONV_SIZE, CONV_WEIGHT_1_FEATURES, CONV_WEIGHT_2_FEATURES])
	conv_bias2 = get_bias_variable([CONV_WEIGHT_2_FEATURES])

	layer_2_weights_histogram = tf.histogram_summary("conv_weights2", conv_weights2)
	layer_2_bias_histogram = tf.histogram_summary("conv_bias2", conv_bias2)

	convolution2 = tf.nn.tanh(conv2d(convolution1, conv_weights2) + conv_bias2)

	fully_connected_weights1 = get_weight_variable([CONV_2_OUTPUT_SIZE, FC_LAYER_1_WEIGHTS])
	fully_connected_bias1 = get_bias_variable([FC_LAYER_1_WEIGHTS])

	layer_3_weights_histogram = tf.histogram_summary("fully_connected_weights1", fully_connected_weights1)
	layer_3_bias_histogram = tf.histogram_summary("fully_connected_bias1", fully_connected_bias1)

	conv2_flat = tf.reshape(convolution2, [-1, CONV_2_OUTPUT_SIZE])
	fully_connected_output1 = tf.nn.tanh(tf.matmul(conv2_flat, fully_connected_weights1) + fully_connected_bias1)

	keep_prob = tf.placeholder(tf.float32)
	fully_connected1_drop = tf.nn.dropout(fully_connected_output1, keep_prob)

	fully_connected_weights2 = get_weight_variable([FC_LAYER_1_WEIGHTS, OUTPUT_SIZE])
	fully_connected_bias2 = get_bias_variable([OUTPUT_SIZE])

	layer_4_weights_histogram = tf.histogram_summary("fully_connected_weights2", fully_connected_weights2)
	layer_4_bias_histogram = tf.histogram_summary("fully_connected_bias2", fully_connected_bias2)

	tf_output = tf.matmul(fully_connected1_drop, fully_connected_weights2) + fully_connected_bias2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf_output, training_output))
	tf.histogram_summary("cross_entropy", cross_entropy)

	#learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_LEARNING_RATE_EVERY_N, DECAY_RATE, staircase=True)
	#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy, global_step=global_step)

	correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(tf_output),1), tf.argmax(training_output,1))
	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.histogram_summary("accuracy", accuracy)

	# Allows saving the state of all tf variables
	saver = tf.train.Saver()
	
	merged_summary_op = tf.merge_all_summaries()
	
	sess = tf.Session()

	#This is necessary to ensure compatibility with two different versions of tensorflow (windows and ubuntu)
	try:
		sess.run(tf.global_variables_initializer())
	except AttributeError as error:
		sess.run(tf.initialize_all_variables())
	
	summary_writer = tf.train.SummaryWriter(GRAPH_LOGS_SAVE_FILE_PATH, graph=sess.graph)

	print("---")
	if should_use_save_data:
		#Try load the weights and biases from when the network was last run
		try:
			saver.restore(sess, MODEL_SAVE_FILE_PATH)
			print("TensorFlow model loaded from last session.")
		except ValueError as error:
			print("Could not load a TensorFlow model from the last session because none was found.")
	else:
		print("Previous save data not loaded! If you wish to load the previous save data run: python3 main.py True")
	print("---")
	print("Network training starting!")

	train_input_batch, train_output_batch = convert_training_to_batch(training_data, NUMBER_OF_BATCHES)
	test_input_batch, test_output_batch = convert_training_to_batch(testing_data, NUMBER_OF_BATCHES)
	feed_dict_train = []
	feed_dict_train_keep_all = []
	feed_dict_test = []
	for i in range (NUMBER_OF_BATCHES):
		train_input_batch[i] = one_hot_input_batch(train_input_batch[i])
		test_input_batch[i] = one_hot_input_batch(test_input_batch[i])
		feed_dict_train.append({training_input: train_input_batch[i], training_output: train_output_batch[i], keep_prob: KEEP_SOME_PROBABILITY})
		feed_dict_train_keep_all.append({training_input: train_input_batch[i], training_output: train_output_batch[i], keep_prob: KEEP_ALL_PROBABILITY})
		feed_dict_test.append({training_input: test_input_batch[i], training_output: test_output_batch[i], keep_prob: KEEP_ALL_PROBABILITY})


	for i in range(NUMBER_OF_BATCHES_TO_TRAIN_ON):
		for j in range(TRAINING_ITERATIONS):
			print("-")
			print("Batch number: " + str(i+1) + "/" + str(NUMBER_OF_BATCHES_TO_TRAIN_ON) + " Training step: " + str(j+1) + "/" + str(TRAINING_ITERATIONS) +  " Global step: " + str(sess.run(global_step)))
			entropy, _, train_step_accuracy = sess.run([cross_entropy,train_step, accuracy], feed_dict=feed_dict_train[i])
			print("Entropy: " + str(entropy))
			print("Training Step Result Accuracy: " + str(train_step_accuracy))
			train_input_batch[i], train_output_batch[i] = shuffle(train_input_batch[i], train_output_batch[i])
			feed_dict_train[i]={training_input: train_input_batch[i], training_output: train_output_batch[i], keep_prob: KEEP_SOME_PROBABILITY}		
		print("Testing Accuracy on random testing batch: " + str(sess.run(accuracy, feed_dict=feed_dict_test[i])))

	debug_outputs = sess.run(tf_output, feed_dict=feed_dict_train_keep_all[0])
	print_debug_outputs(DEBUG_PRINT_SIZE, train_output_batch[0], debug_outputs)

	summary_str = sess.run(merged_summary_op, feed_dict=feed_dict_train_keep_all[0])
	summary_writer.add_summary(summary_str, 0)

	print("NN training complete, moving on to testing.")

	training_accuracy = []
	testing_accuracy = []
	for i in range(NUMBER_OF_BATCHES_TO_TRAIN_ON):
		training_accuracy.append(sess.run(accuracy, feed_dict=feed_dict_train_keep_all[i]))
		testing_accuracy.append(sess.run(accuracy, feed_dict=feed_dict_test[i]))
	training_average = sum(training_accuracy)/len(training_accuracy)
	testing_average = sum(testing_accuracy)/len(testing_accuracy)
	print_accuracy_percentage(training_average, testing_average)

	save_path = saver.save(sess, MODEL_SAVE_FILE_PATH)
	print("TensorFlow model saved in file: %s" % save_path)

	print("Run the following command to see the graphs produced from this training:")
	print("tensorboard --logdir=" + GRAPH_LOGS_SAVE_FILE_PATH)


def use_network(input):
	input = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
	training_input = tf.placeholder(tf.float32, [None, INPUT_SIZE])
	training_output = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
	keep_prob = tf.placeholder(tf.float32)
	global_step = tf.Variable(0, trainable=False)

	conv_weights1 = get_weight_variable([CONV_SIZE, CONV_SIZE, CONV_WEIGHT_1_INPUT_CHANNELS, CONV_WEIGHT_1_FEATURES])
	conv_bias1 = get_bias_variable([CONV_WEIGHT_1_FEATURES])

	input_image = tf.reshape(training_input, [-1, BOARD_SIZE, BOARD_SIZE, COLOUR_CHANNELS_USED])

	convolution1 = tf.nn.tanh(conv2d(input_image, conv_weights1) + conv_bias1)

	conv_weights2 = get_weight_variable([CONV_SIZE, CONV_SIZE, CONV_WEIGHT_1_FEATURES, CONV_WEIGHT_2_FEATURES])
	conv_bias2 = get_bias_variable([CONV_WEIGHT_2_FEATURES])

	convolution2 = tf.nn.tanh(conv2d(convolution1, conv_weights2) + conv_bias2)

	fully_connected_weights1 = get_weight_variable([CONV_2_OUTPUT_SIZE, FC_LAYER_1_WEIGHTS])
	fully_connected_bias1 = get_bias_variable([FC_LAYER_1_WEIGHTS])

	conv2_flat = tf.reshape(convolution2, [-1, CONV_2_OUTPUT_SIZE])
	fully_connected_output1 = tf.nn.tanh(tf.matmul(conv2_flat, fully_connected_weights1) + fully_connected_bias1)

	keep_prob = tf.placeholder(tf.float32)
	fully_connected1_drop = tf.nn.dropout(fully_connected_output1, keep_prob)

	fully_connected_weights2 = get_weight_variable([FC_LAYER_1_WEIGHTS, OUTPUT_SIZE])
	fully_connected_bias2 = get_bias_variable([OUTPUT_SIZE])

	tf_output = tf.matmul(fully_connected1_drop, fully_connected_weights2) + fully_connected_bias2

	# Allows saving the state of all tf variables
	saver = tf.train.Saver()

	sess = tf.Session()

	# This is necessary to ensure compatibility with two different versions of tensorflow (windows and ubuntu)
	try:
		sess.run(tf.global_variables_initializer())
	except AttributeError as error:
		sess.run(tf.initialize_all_variables())

	saver.restore(sess, MODEL_SAVE_FILE_PATH)

	test_input = []
	test_input.append(input)
	test_input_batch = one_hot_input_batch(test_input)
	feed_dict_test = {training_input: test_input_batch, keep_prob: KEEP_ALL_PROBABILITY}

	output = sess.run(tf_output, feed_dict=feed_dict_test)
	print (output)
	winner = get_winner(output[0])
	print (winner)
	difference = get_difference(output[0])
	print (difference)
	scale_difference(difference)


def get_winner(output):
	max_value = max(output)
	winner = numpy.where(output == max_value)
	if winner == 0:
		return -1
	else:
		return 1

def get_difference(output):
	return abs(output[0] - output[1])

def scale_difference(difference):
	return 1

def print_debug_outputs(amount, train_output_batch, debug_outputs):
	print("---")
	print("Debugging/printing random outputs from tensorflow compared to the actual outputs...")
	for i in range(amount):
		random_move_index = randrange(0, len(train_output_batch))
		print("index of move in output array: " + str(random_move_index))
		print("tf output: " + str(debug_outputs[random_move_index]))
		print("Actual output: " + str(train_output_batch[random_move_index]))
	print("---")

def print_accuracy_percentage(training_accuracy, testing_accuracy):
	training_accuracy = "%.2f" % (training_accuracy * 100)
	testing_accuracy = "%.2f" % (testing_accuracy * 100)
	print("-----")
	print("Training Accuracy: " + str(training_accuracy) + "%")
	print("Testing Accuracy: " + str(testing_accuracy) + "%")
	print("-----")

def shuffle(batch_input, batch_ouput):
	#combine the lists (so they keep the same shuffle order), shuffle them, then split them
	#zipping will make the two lists [a, b] and [1, 2] = [(a, 1), (b, 2)]
	combined_batch = list(zip(batch_input, batch_ouput))
	random.shuffle(combined_batch)
	#[:] just essentially casts the tuple result from zip into the same list variables we already used
	batch_input[:], batch_ouput[:] = zip(*combined_batch)
	return batch_input, batch_ouput

if __name__ == '__main__':
	use_network(0)
