#! /usr/bin/env python3
import os.path
from copy import deepcopy
from sys import argv

from board import board as brd


def expand_dirs(path, arr=None):
	if arr is None:
		arr = []
	if os.path.isdir(path):
		for f in os.listdir(path):
			expand_dirs(path + os.sep + f, arr)
	elif os.path.isfile(path) and os.path.splitext(path)[1] == '.psq':
		arr.append(path)
	return arr


def parse_training_file(path):
	with open(path) as f:
		ls = f.readlines()

	ls = ls[1:-3]
	ls = [l.strip() for l in ls]
	ls = [l.split(',')[:2] for l in ls]
	return [(int(m[0]) - 1, int(m[1]) - 1) for m in ls]


def simulate(moves, should_print=False):
	board = [[0 for j in range(brd.BOARD_SIZE)] for i in range(brd.BOARD_SIZE)]
	all_boards = [board]
	p = -1
	for x, y in moves:
		assert board[x][y] == 0
		board[x][y] = p
		all_boards.append(deepcopy(board))
		if should_print:
			brd.print_board(board)
		winner = brd.decide_winner(board)
		if winner != 0:
			return (winner, all_boards)
		p = -p
	raise ValueError('Winner still not determined after all moves have been made.')


def simulate(moves, should_print=False):
	board = brd.Board()
	all_boards = [board.board]
	p = -1
	for x, y in moves:
		assert board.board[x][y] == 0
		board.board[x][y] = p
		all_boards.append(deepcopy(board.board))
		if should_print:
			board.print_board()
		winner = board.decide_winner()
		if winner != 0:
			return winner, all_boards
		p = -p
	raise ValueError('Winner still not determined after all moves have been made.')

def processTrainingData(paths):
	trainingData = []
	for path in paths:
		pathData = []
		for filename in expand_dirs(path):
			print('processing file', filename)
			moves = parse_training_file(filename)
			try:
				winner, boards = simulate(moves, should_print=False)
				pathData.extend((b, winner) for b in boards)
			except ValueError as error:
				print ("Caught the following error for file: ", filename, " Error: ", error)
		if pathData == []:
			#TODO: make it so if the winner is not determined this message changes
			print("Can't read/find file ", path)
		else:
			trainingData.append(pathData)
	return trainingData

if __name__ == '__main__':
	processTrainingData(argv[1:])
