"""
Contains the logic required to run the bot against a board, and make a move.
"""
from copy import deepcopy

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
		self.neural_network = Neural_Network()
		self.node = None
		self.player_int = 0

	def make_move(self, brd: Board) -> MoveStruct:
		if self.node is None:
			board = deepcopy(brd)
			if board.get_last_move() is None:
				self.player_int = -1
			else:
				self.player_int = 1
			self.node = mc.Node((None, None), board, self.neural_network, self.player_int)
		else:
			last_move = brd.get_last_move()
			if last_move is not None:
				node_found = False
				for child in self.node.children:
					if child.get_move() == last_move:
						node_found = True
						self.node = child
						break
				if not node_found:
					board = deepcopy(brd)
					self.node = mc.Node(last_move, board, self.neural_network, self.player_int)

		self.node = self.node.select()
		x, y = self.node.get_move()
		return x, y
