"""
nn : Board -> [-1 to 1]
moves : Board -> list[Board]
moves(b) = [] Person to move last.
"""
from typing import List

from board import Board


def nn(board) -> int:
	# Will return value between -1 and 1. Needs to hook up with the neural network when it's ready.
	pass


def moves(board: Board) -> List[Board]:
	"""
	Returns the number of moves that can be made on the board currently, as boards.
	"""
	moves = board._possible_moves
	boards = []
	for move in moves:
		boards.append()
	return board._possible_moves


def winning_moves(b, depth) -> List:
	# All possible moves that can be made on the current board.
	cs = moves(b)
	if depth == 0:
		# Returns a list of moves that have a higher than 0.7 probability of being in our favour.
		return [c for c in cs if nn(c) > 0.7]
	# No moves can be made anyway, so we can't give a list of winning moves.
	if len(cs) == 0:
		return []

	rs = []
	"""
	For all winning moves, if the move after could lead to a winning move for the other player,
	remove it from the list of winning moves.
	"""
	for c in cs:
		ms = winning_moves(b, depth - 1)
		if len(ms) == 0:
			rs.append(c)
	return rs
