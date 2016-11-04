import tensorflow as tf
import math
from training_data import process_training_data, get_files, get_test_files

'''
-make the numbers make sense mathematically, and make it compile with said numbers
-figure out trunticated normal should we use it???
-figure out what our output data should look like when we feed to to tensorflow.
'''

#TODO: this should be got from the board file
# Width and Height of the board
BOARD_SIZE = 20

LEARNING_RATE = 0.1
#The rate at which neurons are kept after learning
KEEP_PROBABILITY = 0.5

TRAINING_DATA_FILE_COUNT = 500
TEST_DATA_FILE_COUNT = 50

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
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	#A 4-D Tensor with shape [batch, height, width, channels]
	#ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
	#strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
  return tf.nn.max_pool(x, ksize=[1, 1, 1, 1], 
  		strides=[1, 1, 1, 1], padding='SAME')  


#ARGMAX SHOULD JUST BE LOOKING FOR 111 OR 000
#even_count = -1 winner
#odd_count = 1 winner
def get_winner_from_output(output, print_counts):
	odd_count = 0
	even_count = 0
	for i in range(len(output)):
		if output[i] % 2 == 0:
			even_count += 1
		elif output[i] % 2 != 0:
			odd_count += 1
	if print_counts:
		print("Chance the nn thinks it might be 1: " + str(odd_count) + "/20")
		print("Chance the nn thinks it might be -1: " + str(even_count) + "/20")
	if odd_count < even_count:
		return -1
	elif even_count < odd_count:
		return 1
	else:
		#have to do something if the nn is 50:50
		return -1


def conv_network():
	print("Convolutional Neural Network training beginning...")
	print("If you see any 'can't read file' error messages below, please ignore them for now, this is normal behaviour in this early build")
	print("------------------------------------------------")

	# Get training data
	training_data = get_training_data()
	testing_data = get_test_data()

	# Set up placeholders for input and output
	training_input = tf.placeholder(tf.float32, [BOARD_SIZE * BOARD_SIZE])
	training_output = tf.placeholder(tf.float32, [BOARD_SIZE * BOARD_SIZE])

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
	#here we'll have 40 features for each 5x5 patch and 20 input channels
	conv_weights2 = get_weight_variable([5, 5, 20, 40])
	conv_bias2 = get_bias_variable([40])

	convolution2 = tf.nn.relu(conv2d(pool1, conv_weights2) + conv_bias2)
	pool2 = max_pool_2x2(convolution2)

	fully_connected_weights1 = get_weight_variable([5 * 5 * 40, 1000])
	fully_connected_bias1 = get_bias_variable([1000])

	pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 40])
	fully_connected_output1 = tf.nn.relu(tf.matmul(pool2_flat, fully_connected_weights1) + fully_connected_bias1)

	#get rid of some neurons
	keep_prob = tf.placeholder(tf.float32)
	fully_connected1_drop = tf.nn.dropout(fully_connected_output1, keep_prob)

	fully_connected_weights2 = get_weight_variable([1000, 25])
	fully_connected_bias2 = get_bias_variable([25])

	output = tf.nn.softmax(tf.matmul(fully_connected1_drop, fully_connected_weights2) + fully_connected_bias2, 0)
	output = tf.reshape(output, [-1])
	#output = tf.sigmoid(output)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, training_output))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	#correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(training_output,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	print_counter = 0
	# For each game
	for i in range (len(training_data)):
		# For each move in the game
		for j in range (len(training_data[i])):
			batch_input = hotfoot(training_data[i][j][0])
			batch_output = hotfoot(transform_training_output_for_tf(training_data[i][j][1]))
			sess.run(train_step, feed_dict={training_input: batch_input, training_output: batch_output, keep_prob: KEEP_PROBABILITY})
			print_counter = print_counter + 1
			if print_counter % 1000 == 0:
				printable_output = sess.run(output, feed_dict={training_input: batch_input, training_output: batch_output, keep_prob: 1})
				print("Game number: " + str(i))
				print("Output from network for assumed winner (200 or over = 1, less than 200 = -1:")
				print(sess.run(tf.argmax(printable_output, 0)))
				print("Output from training data:")
				print(training_data[i][j][1])
				print("********************************")

	print("NN training complete, moving on to testing.")

	#TODO: move this to its own method
	print_counter = 0
	size = count_moves(testing_data)
	correct = 0
	for i in range(0, len(testing_data)):
		for j in range(0, len(testing_data[i])):
			test_input = hotfoot(testing_data[i][j][0])
			test_output = hotfoot(transform_training_output_for_tf(testing_data[i][j][1]))
			comparable_output = sess.run(output, feed_dict={training_input: test_input, training_output: test_output, keep_prob: 1})
			comparable_output = sess.run(tf.argmax(comparable_output, 0))
			if comparable_output <= 200:
				comparable_output = -1
			elif comparable_output > 200:
				comparable_output = 1
			if print_counter % 50 == 0:
				print("output: " + str(comparable_output) + ", winner: " + str(testing_data[i][j][1]))
			if comparable_output == testing_data[i][j][1]:
				correct += 1

	print("correct: %s" % (correct))
	print("number: %s" % size)
	accuracy = (correct / size) * 100
	print("%s percent" % (accuracy))

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
            printable_output = sess.run(model, feed_dict={input: batch_input, correct: batch_output})
            print(printable_output[0])
            print("Output from training data:")
            print(transform_training_output_for_tf(training_data[i][j][1])[0][0])
            print("********************************")
 

#Tensorflow cannot output a different shape than its input (which is a 20x20 board)
#so we have to do this so we can compare the actual training ouput with the tensorflow output
'''
	I've altered this method so that the bottom 200 tiles are filled to signify player 1 winning
	and the top 200 rows filled to signify -1 winning.

	This is done so we can compare out output with argmax in tensorflow. 
	Argmax will return the max index in the 0 dimension (which is the only dimension since our input is hotfooted).
	If this index is 200 or after it means it thinks player 1 will win if it's below 200 then player -1

	^^^This doesn't actually work and neither does filling the first and second element to represent the winner 
	(because most the time they're 0 but the output data seems to just have random elements in the 400 long array flash on and off
	 ...but most of the elements are 0)

	 TODO: we need to think of a stratgy for making our output training data meaningful to our tensorflow program.
	 It'll probably be smart to talk to Radu about this.
'''
def transform_training_output_for_tf(actual_training_output):
	output = []
	count = 0
	for i in range(BOARD_SIZE):
		output.append([])
		for j in range(BOARD_SIZE):
			count += 1
			if count > 200 and actual_training_output == 1:
				output[i].append(1)
			elif count <= 200 and actual_training_output == -1:
				output[i].append(1)
			else:
				output[i].append(0)
	return output

def hotfoot(input):
	output = []
	for i in input:
		for j in i:
			output.append(j)
	return output

if __name__ == '__main__':
    conv_network()