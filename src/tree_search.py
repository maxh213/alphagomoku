"""
nn : Board -> [-1 to 1]
moves : Board -> list[Board]
moves(b) = [] Person to move last.
"""
from typing import List

from board import Board


def nn(board) -> int:
	# Will return value between -1 and 1. Needs to hook up with the neural network when it's ready.
	return 1


def moves(board: Board) -> List[Board]:
	"""
	Returns the number of moves that can be made on the board currently, as boards.
	"""
	return board.get_possible_moves()


def winning_moves(board: Board, depth: int, should_print: bool=False) -> List[Board]:
	"""
	Searches the board for winning moves to the given depth.
	Total number of moves checked = reduce(lambda x, y: x + len(moves) - i, range(i+1))
	"""
	# All possible moves that can be made on the current board.
	cs = moves(board)

	# No moves can be made anyway, so we can't give a list of winning moves.
	if len(cs) == 0:
		return []

	if depth == 0:
		# Returns a list of moves that have a higher than 0.7 probability of being in our favour,
		# Or will literally win the game.
		return [c for c in cs if nn(c) > 0.7 or board.decide_winner() != 0]

	rs = []
	"""
	For all winning moves, if the move after could lead to a winning move for the other player,
	remove it from the list of winning moves.
	"""
	for i, c in enumerate(cs):
		if should_print:
			print('%d/%d' % (i, len(cs)))
		ms = winning_moves(c, depth - 1)
		if len(ms) == 0:
			rs.append(c)
	return rs

def main():
	board = Board()
	# for i in range(20):
	# 	board.move(i, 0, board._next_player)

	m = winning_moves(board, 2, should_print=True)

	print(len(m))

	# boards = board.get_possible_moves()
	# while len(boards) > 0:
	# 	board.print_board()
	# 	board = boards[0]
	# 	boards = board.get_possible_moves()


if __name__ == '__main__':
	main()