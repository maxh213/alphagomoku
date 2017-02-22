import matplotlib.pyplot as plt
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
	network = Neural_Network()
	board = parse_training_file(FILE_LOCATION)
	inp = simulate(board)
	# Random board
	instance = inp[0][30]
	b = Board(instance)
	outputs = gather_outputs(b, network)
	b.print_board()
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

	# Ensure there is some value for every cell on the board
	#if len(nn_outputs) < 400:
	#	for i in range(len(nn_outputs), 400):
	#		nn_outputs.append(0)

	# Debugging print
	for output in nn_outputs:
		print(output)

	return nn_outputs

# Draw the heatmap
# Code inspired by http://matplotlib.org/examples/pylab_examples/pcolor_demo.html
# ! - Above deprecated, see below - !
# Code inspired by http://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set
# ---
# Code inspired by http://stackoverflow.com/questions/36393929/python-matplotlib-making-heat-map-out-of-tuples-x-y-value
def draw_graph(outputs):
	# Generate some test data
	# x = np.random.randn(8873)
	# y = np.random.randn(8873)
	# Generate board's edges
	x = []
	y = []
	for i in range(1, 21):
		for j in range(1, 21):
			x.append(j)
			y.append(i)

	print(x)
	print(y)


	heatmap, _, _ = np.histogram2d(x, y, weights=outputs, bins = 20)

	plt.clf()
	plt.imshow(heatmap)
	plt.show()

	# heatmap, xedges, yedges = np.histogram2d(x, y, bins=20)
	# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

	# plt.clf()
	# plt.imshow(heatmap.T, extent=extent, origin='lower')
	# plt.show()

if __name__ == "__main__":
    main()
