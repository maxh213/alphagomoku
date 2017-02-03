#! /usr/bin/env python3
import glob
import pickle
from sys import argv
from typing import List, Tuple, Any
from board import Board, BoardStruct, MovesStruct
from os.path import isfile

"""
Training data is represented as a list of moves/boards, and the winner for that game.
"""
TrainingDataStruct = Tuple[List[BoardStruct], int]

"""
All files from freestyle 1 and 3, and only the first 500 from freestyle 2.
_TEST_DATA_FILES uses the rest of freestyle2.
"""
_TRAINING_DATA_FILES = glob.glob("../resources/training/freestyle/freestyle1/*.psq")
_TRAINING_DATA_FILES += glob.glob("../resources/training/freestyle/freestyle3/*.psq")
_TRAINING_DATA_FILES += glob.glob("../resources/training/freestyle/freestyle2/*.psq")[:500]
_TRAINING_DATA_FILES += glob.glob("../resources/training/freestyle/new_training_data/*.psq")[:2467] #2467 * 2 is the number of files in the folder

"""
All but the first 500 files from freestyle 2.
_TRAINING_DATA_FILES uses the other 500.
"""
_TEST_DATA_FILES = glob.glob("../resources/training/freestyle/freestyle2/*.psq")[500:]
_TEST_DATA_FILES += glob.glob("../resources/training/freestyle/new_training_data/*.psq")[2467:]

_TRAINING_DATA_SAVE_PATH = "save_data/training_data.pckl"
_TESTING_DATA_SAVE_PATH = "save_data/testing_data.pckl"


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


def _save_data(file_path: str, data: List[TrainingDataStruct]) -> None:
	'''
	Takes a list of pre-parsed training data, and stores it at the given path, for quicker access.
	'''
	file = open(file_path, 'wb')
	pickle.dump(data, file)
	file.close()


def _load_data(file_path: str) -> List[TrainingDataStruct]:
	'''
	Takes a file path, and returns the list of training data stored there.
	'''
	f = open(file_path, 'rb')
	data = pickle.load(f)
	f.close()
	return data


def _load_or_parse_data(parse_paths: List[str], save_path: str, file_count: int=None) -> List[TrainingDataStruct]:
	'''
	If the given save path already exists, attempts to extract a list of training data from it.
	Otherwise, the parse paths are used to generate data, which is stored at the given save path, and then returned.

	The file count is ignored when saving data.
	This also means that all of the files given by parse_paths will be parsed.

	:param parse_paths: A list of file paths for training data to be parsed.
	:param save_path: A path that may or may not contain pre-parsed training data; written to, in the latter case.
	:param file_count: The amount of training data to return.
	:return: Parsed training data, one way or another.
	'''
	if isfile(save_path):
		data = _load_data(save_path)
	else:
		data = process_training_data(parse_paths)
		_save_data(save_path, data)

	if file_count is None:
		return data
	return data[:file_count]


def get_training_data(file_count: int=None):
	return _load_or_parse_data(_TRAINING_DATA_FILES, _TRAINING_DATA_SAVE_PATH, file_count)


def get_test_data(file_count: int=None):
	return _load_or_parse_data(_TEST_DATA_FILES, _TESTING_DATA_SAVE_PATH, file_count)


if __name__ == '__main__':
	if len(argv) > 1:
		process_training_data(argv[1:], should_print=True)
	else:
		process_training_data(_TRAINING_DATA_FILES, should_print=True)
