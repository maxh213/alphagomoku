import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import numpy.random

from neuralnetwork.training_data import parse_training_file, simulate
from gomokuapp.board import Board
from treesearch.monte_carlo import Neural_Network

from copy import deepcopy

import sys

FILE_LOCATION = "resources/training/training_data_max/2x1-31(98).psq"

# Generate network input, determine network output, create heatmap
def main(file_path=FILE_LOCATION, game_number=16):
	network = Neural_Network()

	board = parse_training_file(file_path)
	inp = simulate(board)
	instance = inp[0][int(game_number)] # Some move within a game
	b = Board(instance)

	# Gather neural network outputs for each possible move on board
	outputs = gather_outputs(b, network)
	b.print_board()

	# Call with additional parameter False to prevent PDF output
	draw_graph(outputs)


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
def draw_graph(outputs, should_save=True):
	# Generate the coordinates
	x = []
	y = []
	for i in range(0, 20):
		for j in range(0, 20):
			x.append(j)
			y.append(i)

	heatmap, _, _ = np.histogram2d(x, y, weights=outputs, bins = 20)

	f = plt.figure()

	plt.clf()
	plt.imshow(heatmap)

	# Style
	plt.set_cmap("hot")
	plt.title("Heatmap of Neural Network Output")
	plt.xlabel("X Coordinate on Board")
	plt.ylabel("Y Coordinate on Board")
	plt.colorbar().set_label("Neural Network Output")

	# Bespoke config
	plt.gca().invert_yaxis()
	plt.xticks(np.arange(min(x), max(x)+1, 2.0))
	plt.yticks(np.arange(min(y), max(y)+1, 2.0))
	plt.show()

	if should_save:
		f.savefig("resources/heatmaps/heatmap.pdf")


if __name__ == "__main__":
	# The user can supply a file location and game count to generate a heatmap for
	# Otherwise, default values will be used
	if len(sys.argv) == 3:
		path = sys.argv[1]
		game = sys.argv[2]
		try:
			main(path, game)
		except IOError as fnf:
			print("%s is not a valid file location." % (path))
		except Exception as e:
			print("Please enter a valid game count.")
			print(e)
	else:
		main()
