"""
Contains the logic required to run the bot against a board, and make a move.
"""

import tree_search as ts
from board import Board

TREE_SEARCH_DEPTH = 2


def make_move(brd: Board):
	x, y = ts.winning_moves(brd, TREE_SEARCH_DEPTH)[0]
	return x, y