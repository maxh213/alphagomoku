"""
nn : Board -> [-1 to 1]
moves : Board -> list[Board]
moves(b) = [] Person to move last.
"""
from typing import List


def nn(board) -> int:
	# Will return value between -1 and 1. Needs to hook up with the neural network when it's ready.
	pass


def moves(b) -> List:
	pass


def winning_moves(b, depth) -> List:
	rs = []
	cs = moves(b)
	if depth == 0:
		return [c for c in cs if nn(c) > 0.7]
	if len(cs) == 0:
		return []
	for c in cs:
		ms = winning_moves(b, depth - 1)
		if len(ms) == 0:
			rs.append(c)
	return rs
