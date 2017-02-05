"""
Contains the logic required to run the bot against a board, and make a move.
"""

import treesearch as ts
from gomokuapp.board import Board, MoveStruct
from treesearch import monte_carlo as mc
from treesearch.monte_carlo import Neural_Network

TREE_SEARCH_DEPTH = 2


def make_move(brd: Board) -> MoveStruct:
	x, y = ts.winning_moves(brd, TREE_SEARCH_DEPTH)[0]
	return x, y


class Computer:
	def __init__(self):
		self.board = Board()
		self.neural_network = Neural_Network()
		# Todo: Come up with a better way to start the nodes off.
		self.node = mc.Node((None, None), self.board, self.neural_network)
		

	def make_move(self, brd: Board) -> MoveStruct:
		last_move = brd.get_last_move()

		if last_move is not None:
			x, y = last_move
			self.board.move(x, y, self.board.get_next_player())

		self.node = self.node.select()
		x, y = self.node.get_move()
		self.board.move(x, y, self.board.get_next_player())
		return x, y
