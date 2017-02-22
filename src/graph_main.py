import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import numpy.random

from neuralnetwork.training_data import parse_training_file, simulate
from gomokuapp.board import Board
from treesearch.monte_carlo import Neural_Network

from copy import deepcopy

# Change as appropriate
# TO DO: Parameterise
FILE_LOCATION = "resources/training/training_data_max/2x1-31(98).psq"

# Generate network input, determine network output, create heatmap
def main():
	# True: Save heatmap to PDF
	# False: Do not save heatmap to PDF
	should_save = True

	network = Neural_Network()

	board = parse_training_file(FILE_LOCATION)
	inp = simulate(board)
	instance = inp[0][30] # Some move within a game
	b = Board(instance)

	# Gather neural network outputs for each possible move on board
	outputs = gather_outputs(b, network)
	b.print_board()
	draw_graph(outputs)

	if should_save:
		save_to_PDF()

def save_to_PDF():
	print("hello")

# For a given board, determine the network's output for all adjacent states
def gather_outputs(board, network):
	nn_outputs = []
	moves = board.get_possible_moves()
	played = board.get_played_moves()
	total = moves + played
	neww = sorted(total)
	for move in neww:
		if move not in played:
			new_board = deepcopy(board)
			valid = new_board.move(move[0], move[1], board.get_next_player())
			if valid:
				nn_outputs.append(network.nn(new_board, new_board.get_next_player()))
			else:
				# This should never happen
				assert False
		else:
			nn_outputs.append(0)

	return nn_outputs

# Draw the heatmap
def draw_graph(outputs):
	# Generate the coordinates
	x = []
	y = []
	for i in range(1, 21):
		for j in range(1, 21):
			x.append(j)
			y.append(i)

	heatmap, _, _ = np.histogram2d(x, y, weights=outputs, bins = 20)

	f = plt.figure()

	plt.clf()
	plt.title("Heatmap of Neural Network Output")
	plt.imshow(heatmap)
	plt.show()

	f.savefig("resources/heatmaps/heatmap.pdf")


if __name__ == "__main__":
    main()
