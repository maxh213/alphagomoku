import tensorflow as tf
import math
from random import randrange
from training_data import get_training_data, get_test_data, get_batch, one_hot_input_batch, get_testing_data_save_path, get_training_data_save_path
import pickle

# TODO: this should be got from the board file
# Width and Height of the board
BOARD_SIZE = 20

LEARNING_RATE = 0.03
# The rate at which neurons are kept after learning
KEEP_SOME_PROBABILITY = 0.5
KEEP_ALL_PROBABILITY = 1.0

TRAINING_DATA_FILE_COUNT = 2500
TEST_DATA_FILE_COUNT = 500

MODEL_SAVE_FILE_PATH = "save_data/models/model.ckpt"
GRAPH_LOGS_SAVE_FILE_PATH = "save_data/logs/"

def get_weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def get_bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def neural_network_train(should_use_save_data):
	print("Convolutional Neural Network training beginning...")
	print("If you see any 'can't read file' error messages below, please ignore them for now, this is normal behaviour in this early build")
	print("------------------------------------------------")

	# Get training data
	training_data = get_training_data(TRAINING_DATA_FILE_COUNT)
	testing_data = get_test_data(TEST_DATA_FILE_COUNT)

	# Set up placeholders for input and output
	training_input = tf.placeholder(tf.float32, [None, 400])
	training_output = tf.placeholder(tf.float32, [None, 2])
	keep_prob = tf.placeholder(tf.float32) 	

	layer_1_weights = get_weight_variable([400, 2])
	layer_1_bias = get_bias_variable([2])

	layer_1_weights_histogram = tf.histogram_summary("layer_1_weights", layer_1_weights)
	layer_1_bias_histogram = tf.histogram_summary("layer_1_bias", layer_1_bias)

	layer_1_output = tf.sigmoid(tf.matmul(training_input, layer_1_weights) + layer_1_bias)
	
	#[2, 2] not [400, 2] because the layer above transforms the output
	layer_2_weights = get_weight_variable([2, 2])
	layer_2_bias = get_bias_variable([2])

	layer_2_weights_histogram = tf.histogram_summary("layer_2_weights", layer_2_weights)
	layer_2_bias_histogram = tf.histogram_summary("layer_2_bias", layer_2_bias)

	layer_2_output = tf.nn.sigmoid(tf.matmul(layer_1_output, layer_2_weights) + layer_2_bias)

	#Drop some of the neurons, this is done to try prevent overfitting (the NN outputting the same pattern or output everytime)
	layer_2_dropout_output = tf.nn.dropout(layer_2_output, keep_prob) 

	layer_3_weights = get_weight_variable([2, 2])
	layer_3_bias = get_bias_variable([2])

	layer_3_weights_histogram = tf.histogram_summary("layer_3_weights", layer_3_weights)
	layer_3_bias_histogram = tf.histogram_summary("layer_3_bias", layer_3_bias)

	layer_3_output = tf.nn.sigmoid(tf.matmul(layer_2_dropout_output, layer_3_weights) + layer_3_bias)
	
	tf_output = layer_3_output

	cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf_output, training_output))
	tf.histogram_summary("cross_entropy", cross_entropy)

	train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
	#train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(tf_output,1), tf.argmax(training_output,1))
	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.histogram_summary("accuracy", accuracy)

	# Allows saving the state of all tf variables
	saver = tf.train.Saver()
	
	merged_summary_op = tf.merge_all_summaries()
	
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	
	summary_writer = tf.train.SummaryWriter(GRAPH_LOGS_SAVE_FILE_PATH, graph=sess.graph)

	if should_use_save_data == ['True'] or should_use_save_data == ['true']:
		#Try load the weights and biases from when the network was last run
		try:
			saver.restore(sess, MODEL_SAVE_FILE_PATH)
			print("TensorFlow model restored from last session.")
		except ValueError as error:
			print("Could not load a TensorFlow model from the last session because none was found.")
	else:
		print("Did not load previous save data because you did not pass in a boolean flag saying True. If you wish to load the previous save data run: python3 main.py True")

	print("Network training starting!")

	train_input_batch, train_output_batch = get_batch(training_data)
	train_input_batch = one_hot_input_batch(train_input_batch)
	entropy, _ = sess.run([cross_entropy,train_step], feed_dict={training_input: train_input_batch, training_output: train_output_batch, keep_prob: KEEP_SOME_PROBABILITY})
	print("Entropy: " + str(entropy))
	print("Training Accuracy: " + str(sess.run(accuracy, feed_dict={training_input: train_input_batch, training_output: train_output_batch, keep_prob: KEEP_ALL_PROBABILITY})))
	
	debug_outputs = sess.run(tf_output, feed_dict={training_input: train_input_batch, training_output: train_output_batch, keep_prob: KEEP_ALL_PROBABILITY})
	print_debug_outputs(5, train_output_batch, debug_outputs)

	summary_str = sess.run(merged_summary_op, feed_dict={training_input: train_input_batch, training_output: train_output_batch, keep_prob: KEEP_ALL_PROBABILITY})
	summary_writer.add_summary(summary_str, 0)

	print("NN training complete, moving on to testing.")
	save_path = saver.save(sess, MODEL_SAVE_FILE_PATH)
	print("TensorFlow model saved in file: %s" % save_path)

	test_input_batch, test_output_batch = get_batch(testing_data)
	test_input_batch = one_hot_input_batch(test_input_batch)
	print("Testing Accuracy: " + str(sess.run(accuracy, feed_dict={training_input: test_input_batch, training_output: test_output_batch, keep_prob: KEEP_ALL_PROBABILITY})))

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