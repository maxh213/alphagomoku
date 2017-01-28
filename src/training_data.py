#! /usr/bin/env python3
import glob
from copy import deepcopy
from sys import argv

from typing import List, Tuple
from board import Board, BoardStruct

"""
Training data is represented as a list of moves/boards, and the winner for that game.
"""
TrainingDataStruct = Tuple[List[BoardStruct], int]


"""
Moves are represented as a tuple of x and y coords.
"""
MoveStruct = Tuple[int, int]

MovesStruct = List[MoveStruct]


def parse_training_file(path: str) -> MovesStruct:
	with open(path) as f:
		ls = f.readlines()

	ls = ls[1:-3]
	ls = [l.strip() for l in ls]
	ls = [l.split(',')[:2] for l in ls]
	return [(int(m[0]) - 1, int(m[1]) - 1) for m in ls]


def simulate(moves: MovesStruct, should_print: bool=False) -> TrainingDataStruct:
	board = Board()
	all_boards = [deepcopy(board.board)]
	p = -1
	for x, y in moves:
		assert board.board[x][y] == 0
		board.board[x][y] = p
		all_boards.append(deepcopy(board.board))
		if should_print:
			board.print_board()
		winner = board.decide_winner()
		if winner != 0:
			return all_boards, winner
		p = -p
	raise ValueError('Winner still not determined after all moves have been made.')


def process_training_data(paths: List[str]):
	training_data = []
	for path in paths:
		path_data = []
		print('processing file', path)
		moves = parse_training_file(path)
		try:
			boards, winner = simulate(moves, should_print=False)
			path_data.extend((b, winner) for b in boards)
		except ValueError as error:
			print ("Caught the following error for file: ", path, " Error: ", error)
		if path_data == []:
			#TODO: make it so if the winner is not determined this message changes
			print("Can't read/find file ", path)
		else:
			training_data.append(path_data)
	return training_data


def get_files() -> List[str]:
	"""
	Gets a list of file paths for the training data.
	:rtype: list[str]
	"""
	return glob.glob("../resources/training/freestyle/freestyle1/*.psq")

def get_test_files() -> List[str]:
	return glob.glob("../resources/training/freestyle/freestyle2/*.psq")

if __name__ == '__main__':
	#process_training_data(argv[1:])
	process_training_data(get_files())