import tensorflow as tf
import math
import random
import numpy as np
from training_data import get_training_data, get_test_data, get_batch

# TODO: this should be got from the board file
# Width and Height of the board
BOARD_SIZE = 20

LEARNING_RATE = 1e-4
# The rate at which neurons are kept after learning
KEEP_PROBABILITY = 0.5

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
	training_input = tf.placeholder(tf.float32, [None, BOARD_SIZE, BOARD_SIZE])
	training_output = tf.placeholder(tf.float32, [None, 2])

	fully_connected_weights1 = get_weight_variable([400, 2000])
	fully_connected_bias1 = get_bias_variable([2000])

	fc_weights1_histogram = tf.histogram_summary("fully_connected_weights1", fully_connected_weights1)
	fc_bias1_histogram = tf.histogram_summary("fully_connected_bias1", fully_connected_bias1)

	fc1_flat = tf.reshape(training_input, [-1, 400])
	fully_connected_output1 = tf.nn.relu(tf.matmul(fc1_flat, fully_connected_weights1) + fully_connected_bias1)

	keep_prob = tf.placeholder(tf.float32)
	fully_connected1_drop = tf.nn.dropout(fully_connected_output1, keep_prob)

	fully_connected_weights2 = get_weight_variable([2000, 2])
	fully_connected_bias2 = get_bias_variable([2])

	fc_weights2_histogram = tf.histogram_summary("fully_connected_weights2", fully_connected_weights2)
	fc_bias2_histogram = tf.histogram_summary("fully_connected_bias2", fully_connected_bias2)

	tf_output = tf.nn.softmax(tf.matmul(fully_connected_output1, fully_connected_weights2) + fully_connected_bias2)
	tf_output = tf.sigmoid(tf_output)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf_output, training_output))
	tf.histogram_summary("cross_entropy", cross_entropy)
	#tf.scalar_summary("cross_entropy", cross_entropy)
	train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

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
	print("")

	train_input_batch, train_output_batch = get_batch(training_data)
	entropy, _ = sess.run([cross_entropy,train_step], feed_dict={training_input: train_input_batch, training_output: train_output_batch, keep_prob: KEEP_PROBABILITY})
	print("Entropy: " + str(entropy))
	print("Training Accuracy: " + str(sess.run(accuracy, feed_dict={training_input: train_input_batch, training_output: train_output_batch, keep_prob:1.0})))
	summary_str = sess.run(merged_summary_op, feed_dict={training_input: train_input_batch, training_output: train_output_batch, keep_prob: 1.0})
	summary_writer.add_summary(summary_str, 0)

	print("NN training complete, moving on to testing.")
	save_path = saver.save(sess, MODEL_SAVE_FILE_PATH)
	print("TensorFlow model saved in file: %s" % save_path)

	test_input_batch, test_output_batch = get_batch(testing_data)
	print("Testing Accuracy: " + str(sess.run(accuracy, feed_dict={training_input: test_input_batch, training_output: test_output_batch, keep_prob:1.0})))

	print("Run the following command to see the graphs produced from this training:")
	print("tensorboard --logdir=" + GRAPH_LOGS_SAVE_FILE_PATH)