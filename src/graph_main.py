import matplotlib.pyplot as plt
import numpy as np

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
	draw_graph(outputs)

# For a given board, determine the network's output for all adjacent states
def gather_outputs(board, network):
	nn_outputs = []
	moves = board.get_possible_moves()
	for move in moves:
		new_board = deepcopy(board)
		valid = new_board.move(move[0], move[1], board.get_next_player())
		if valid:
			nn_outputs.append(network.nn(new_board, new_board.get_next_player()))
		else:
			# This should never happen
			assert False
	for output in nn_outputs:
		print(output)

# Draw the heatmap
# Code inspired by http://matplotlib.org/examples/pylab_examples/pcolor_demo.html
def draw_graph(outputs):
	dx, dy = 0.15, 0.05
	# generate 2 2d grids for the x & y bounds
	y, x = np.mgrid[slice(-3, 3 + dy, dy),
	                slice(-3, 3 + dx, dx)]
	z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
	# x and y are bounds, so z should be the value *inside* those bounds.
	# Therefore, remove the last value from the z array.
	z = z[:-1, :-1]
	z_min, z_max = -np.abs(z).max(), np.abs(z).max()

	plt.subplot(2, 2, 1)
	plt.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
	plt.title('pcolor')
	# set the limits of the plot to the limits of the data
	plt.axis([x.min(), x.max(), y.min(), y.max()])
	plt.colorbar()




	plt.subplots_adjust(wspace=0.5, hspace=0.5)

	plt.show()

if __name__ == "__main__":
    main()
