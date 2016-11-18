#! /usr/bin/env python3
import glob
import random
from sys import argv

from typing import List, Tuple
from board import Board, BoardStruct, MovesStruct

"""
Training data is represented as a list of moves/boards, and the winner for that game.
"""
TrainingDataStruct = Tuple[List[BoardStruct], int]

"""
All files from freestyle 1 and 3, and only the first 500 from freestyle 2.
_TEST_DATA_FILES uses the rest of freestyle2.
"""
_TRAINING_DATA_FILES = glob.glob("../resources/training/freestyle/freestyle1/*.psq") \
			+ glob.glob("../resources/training/freestyle/freestyle3/*.psq") + glob.glob(
		"../resources/training/freestyle/freestyle2/*.psq")[:500]

"""
All but the first 500 files from freestyle 2.
_TRAINING_DATA_FILES uses the other 500.
"""
_TEST_DATA_FILES = glob.glob("../resources/training/freestyle/freestyle2/*.psq")[500:]


def parse_training_file(path: str) -> MovesStruct:
	with open(path) as f:
		ls = f.readlines()

	ls = ls[1:-3]
	ls = [l.strip() for l in ls]
	ls = [l.split(',')[:2] for l in ls]
	return [(int(m[0]) - 1, int(m[1]) - 1) for m in ls]


def simulate(moves: MovesStruct, should_print: bool = False) -> TrainingDataStruct:
	board = Board()
	all_boards = [board.get_board()]
	p = -1
	for x, y in moves:
		assert board.move(x, y, p)
		all_boards.append(board.get_board())
		if should_print:
			board.print_board()
		winner, _ = board.decide_winner()
		if winner != 0:
			return all_boards, winner
		p = -p
	raise ValueError('Winner still not determined after all moves have been made.')


def process_training_data(paths: List[str], should_print=False):
	training_data = []
	for path in paths:
		path_data = []
		if should_print:
			print('processing file', path)
		moves = parse_training_file(path)
		try:
			boards, winner = simulate(moves, should_print=should_print)
			path_data.extend((b, winner) for b in boards)
		except ValueError as error:
			print("Warning: Training data not interpretable: %s. Error: %s" % (path, error))
			continue
		if path_data == []:
			# TODO: make it so if the winner is not determined this message changes
			print("Can't read/find file ", path)
		else:
			training_data.append(path_data)
	return training_data

def get_files() -> List[str]:
	"""
	Gets a list of file paths for the training data.
	:rtype: list[str]
	:return _TRAINING_DATA_FILES
	"""
	files = _TRAINING_DATA_FILES
	return files


def get_test_files() -> List[str]:
	"""
	Returns files to be used for testing the NN.
	:return: _TEST_DATA_FILES
	"""
	return _TEST_DATA_FILES

'''
Training data format:
training_data[0] = first game
training_data[0][1][0] = first move of first game
training_data[0][2][0] = second move of first game
training_data[0][0][1] = winner of first game
training_data[0][1][0][0] = first line of first move of first game
training_data[0][1][0][0][0] = first tile on first line of first move of first game

After the training data has been shuffled training data loses the first index above and they're just random moves from any given game
For example:
training_data[0][0] = a move of a random game
training_data[0][1] = winner of the game which the random move was taken from
training_data[1][0] = another move of a random game
and so on...
'''
def get_training_data(file_count):
	# Obtain files for processing
	files = get_files()
	processed_training_files = process_training_data(files[:file_count])
	return processed_training_files
	#shuffled_training_files = shuffle_training_data(processed_training_files)
	#return shuffled_training_files

'''
	returns the training data in a batch format which can be argmaxed by tensorflow
'''
def get_batch(training_data):
	train_input = []
	train_output = []
	for i in range(len(training_data)):
		for j in range(len(training_data[i])):
			train_input.append(training_data[i][j][0])
			if training_data[i][j][1] == -1:
				train_output.append([100, 0])
			elif training_data[i][j][1] == 1:
				train_output.append([0, 100])
			else:
				train_output.append([0, 0])
	return train_input, train_output


def get_test_data(file_count):
	test_files = get_test_files()
	return process_training_data(test_files[:file_count])


if __name__ == '__main__':
	if len(argv) > 1:
		process_training_data(argv[1:], should_print=True)
	else:
		process_training_data(get_files(), should_print=True)
