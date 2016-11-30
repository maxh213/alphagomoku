import tensorflow as tf
import math
from random import randrange
from training_data import get_training_data, get_test_data
from board import BOARD_SIZE

TRAINING_ITERATIONS = 10
DEBUG_PRINT_SIZE = 5

LEARNING_RATE = 0.003
#The below is only needed for gradient decent
#DECAY_LEARNING_RATE_EVERY_STEP_AMOUNT = math.ceil(TRAINING_ITERATIONS/4)
#LEARNING_RATE_DECAY_AMOUNT = 0.1

# The rate at which neurons are kept after learning
KEEP_SOME_PROBABILITY = 0.7
KEEP_ALL_PROBABILITY = 1.0

#Setting the below to None means load all of them
TRAINING_DATA_FILE_COUNT = None 
TEST_DATA_FILE_COUNT = None

MODEL_SAVE_FILE_PATH = "save_data/models/model.ckpt"
GRAPH_LOGS_SAVE_FILE_PATH = "save_data/logs/"

INPUT_SIZE = BOARD_SIZE ** 2
OUTPUT_SIZE = 2
#You can change the below to be whatever you want, the higher they are the longer it'll take to run though
LAYER_1_WEIGHTS_SIZE = 400
LAYER_2_WEIGHTS_SIZE = 200


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
def convert_training_to_batch(training_data):
	train_input = []
	train_output = []
	for i in range(len(training_data)):
		for j in range(len(training_data[i])):
			# if the move is less than 15 and the game lasts more than 15 moves don't bother
			if not (j < 15 and len(training_data[i]) > 15):
				train_input.append(training_data[i][j][0])
				# If training_data[i][j][1] == -1 then an argmax function would identify the first index 0 as the highest
				# If training_data[i][j][1] == 1 then the argmax function would identify index 1 as the highest
				# Our nn just has to mimic this
				train_output.append([0, training_data[i][j][1]])
	return train_input, train_output


def neural_network_train(should_use_save_data):
	print("Convolutional Neural Network training beginning...")

	print("Loading training data...")
	# Get training data
	training_data = get_training_data(TRAINING_DATA_FILE_COUNT)
	testing_data = get_test_data(TEST_DATA_FILE_COUNT)
	print("Training data loaded!")

	# Set up placeholders for input and output
	training_input = tf.placeholder(tf.float32, [None, INPUT_SIZE])
	training_output = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
	keep_prob = tf.placeholder(tf.float32)
	global_step = tf.Variable(0, trainable=False)

	layer_1_weights = get_weight_variable([INPUT_SIZE, LAYER_1_WEIGHTS_SIZE])
	layer_1_bias = get_bias_variable([LAYER_1_WEIGHTS_SIZE])

	layer_1_weights_histogram = tf.histogram_summary("layer_1_weights", layer_1_weights)
	layer_1_bias_histogram = tf.histogram_summary("layer_1_bias", layer_1_bias)

	layer_1_output = tf.nn.tanh(tf.matmul(training_input, layer_1_weights) + layer_1_bias)
	
	#[2, 2] not [400, 2] because the layer above transforms the output
	layer_2_weights = get_weight_variable([LAYER_1_WEIGHTS_SIZE, LAYER_2_WEIGHTS_SIZE])
	layer_2_bias = get_bias_variable([LAYER_2_WEIGHTS_SIZE])

	layer_2_weights_histogram = tf.histogram_summary("layer_2_weights", layer_2_weights)
	layer_2_bias_histogram = tf.histogram_summary("layer_2_bias", layer_2_bias)

	layer_2_output = tf.nn.tanh(tf.matmul(layer_1_output, layer_2_weights) + layer_2_bias)

	#Drop some of the neurons, this is done to try prevent overfitting (the NN outputting the same pattern or output everytime)
	layer_2_dropout_output = tf.nn.dropout(layer_2_output, keep_prob) 

	layer_3_weights = get_weight_variable([LAYER_2_WEIGHTS_SIZE, OUTPUT_SIZE])
	layer_3_bias = get_bias_variable([OUTPUT_SIZE])

	layer_3_weights_histogram = tf.histogram_summary("layer_3_weights", layer_3_weights)
	layer_3_bias_histogram = tf.histogram_summary("layer_3_bias", layer_3_bias)

	layer_3_output = tf.nn.tanh(tf.matmul(layer_2_dropout_output, layer_3_weights) + layer_3_bias)
	
	tf_output = layer_3_output

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf_output, training_output))
	tf.histogram_summary("cross_entropy", cross_entropy)

	#learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_LEARNING_RATE_EVERY_STEP_AMOUNT, LEARNING_RATE_DECAY_AMOUNT, staircase=True)
	#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy, global_step=global_step)
	#train_step = tf.train.AdadeltaOptimizer(LEARNING_RATE, rho=LEARNING_RATE_DECAY_AMOUNT, epsilon=EPSILON, use_locking=False).minimize(cross_entropy, global_step=global_step)

	correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(tf_output),1), tf.argmax(training_output,1))
	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.histogram_summary("accuracy", accuracy)

	# Allows saving the state of all tf variables
	saver = tf.train.Saver()
	
	merged_summary_op = tf.merge_all_summaries()
	
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	
	summary_writer = tf.train.SummaryWriter(GRAPH_LOGS_SAVE_FILE_PATH, graph=sess.graph)

	if should_use_save_data:
		#Try load the weights and biases from when the network was last run
		try:
			saver.restore(sess, MODEL_SAVE_FILE_PATH)
			print("TensorFlow model restored from last session.")
		except ValueError as error:
			print("Could not load a TensorFlow model from the last session because none was found.")
	else:
		print("Did not load previous save data because you did not pass in a boolean flag saying True. If you wish to load the previous save data run: python3 main.py True")

	print("Network training starting!")

	train_input_batch, train_output_batch = convert_training_to_batch(training_data)
	train_input_batch = one_hot_input_batch(train_input_batch)
	feed_dict_train={training_input: train_input_batch, training_output: train_output_batch, keep_prob: KEEP_SOME_PROBABILITY}
	feed_dict_train_keep_all={training_input: train_input_batch, training_output: train_output_batch, keep_prob: KEEP_ALL_PROBABILITY}

	
	for i in range(TRAINING_ITERATIONS):
		#TODO: IT'S A GOOD IDEA TO SHUFFLE THE TRAINING DATA AFTER EVERY EPOCHE TO AVOID BAIS
		print("Training step: " + str(sess.run(global_step)) + "/" + str(TRAINING_ITERATIONS))
		entropy, _, train_step_accuracy = sess.run([cross_entropy,train_step, accuracy], feed_dict=feed_dict_train)
		print("Entropy: " + str(entropy))
		print("Training Step Result Accuracy: " + str(train_step_accuracy))
	print("Training Accuracy: " + str(sess.run(accuracy, feed_dict=feed_dict_train_keep_all)))
	
	debug_outputs = sess.run(tf_output, feed_dict=feed_dict_train_keep_all)
	print_debug_outputs(DEBUG_PRINT_SIZE, train_output_batch, debug_outputs)

	summary_str = sess.run(merged_summary_op, feed_dict=feed_dict_train_keep_all)
	summary_writer.add_summary(summary_str, 0)

	print("NN training complete, moving on to testing.")
	save_path = saver.save(sess, MODEL_SAVE_FILE_PATH)
	print("TensorFlow model saved in file: %s" % save_path)

	test_input_batch, test_output_batch = convert_training_to_batch(testing_data)
	test_input_batch = one_hot_input_batch(test_input_batch)
	feed_dict_test={training_input: test_input_batch, training_output: test_output_batch, keep_prob: KEEP_ALL_PROBABILITY}
	print("Testing Accuracy: " + str(sess.run(accuracy, feed_dict=feed_dict_test)))

	training_accuracy = sess.run(accuracy, feed_dict=feed_dict_train_keep_all)
	testing_accuracy = sess.run(accuracy, feed_dict=feed_dict_test)
	training_accuracy = "%.2f" % (training_accuracy * 100)
	testing_accuracy = "%.2f" % (testing_accuracy * 100)
	print_accuracy_percentage(training_accuracy, testing_accuracy)


	print("Run the following command to see the graphs produced from this training:")
	print("tensorboard --logdir=" + GRAPH_LOGS_SAVE_FILE_PATH)


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
	print("-----")
	print("Training Accuracy: " + str(training_accuracy) + "%")
	print("Testing Accuracy: " + str(testing_accuracy) + "%")
	print("-----")