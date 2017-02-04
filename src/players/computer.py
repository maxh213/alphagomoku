"""
Contains the logic required to run the bot against a board, and make a move.
"""

import treesearch as ts
from gomokuapp.board import Board, MoveStruct
from treesearch import monte_carlo as mc

TREE_SEARCH_DEPTH = 2


def make_move(brd: Board) -> MoveStruct:
	x, y = ts.winning_moves(brd, TREE_SEARCH_DEPTH)[0]
	return x, y


class Computer:
	def __init__(self):
		self.board = Board()
		# Todo: Come up with a better way to start the nodes off.
		self.node = mc.Node((None, None), self.board)

	def make_move(self, brd: Board) -> MoveStruct:
		last_move = brd.get_last_move()

		if last_move is not None:
			x, y = last_move
			self.board.move(x, y, self.board.get_next_player())

		self.node = self.node.select()
		return self.node.get_move()
