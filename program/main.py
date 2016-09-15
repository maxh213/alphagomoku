#! /usr/bin/env python3
import board as brd
from copy import deepcopy
from sys import argv

CHAR_PLAYER_1 = 'X'
CHAR_PLAYER_2 = 'Y'
# Char that represents unclaimed territory
CHAR_NEUTRAL = '-'
PLAYER_CODES = [CHAR_PLAYER_1, CHAR_NEUTRAL, CHAR_PLAYER_2]

def convertPlayerChar(player):
	assert -1 <= player <= 1, "Invalid board cell contents"
	return PLAYER_CODES[player + 1]

def parseTrainingFile(path):
	with open(path) as f:
		ls = f.readlines()

	ls = ls[1:-3]
	ls = [l.strip() for l in ls]
	ls = [l.split(',')[:2] for l in ls]
	return [(int (m[0]) - 1,int (m[1]) - 1) for m in ls]

def simulate(moves, shouldPrint=False):
	board = [[0 for j in range(brd.BOARD_SIZE)]for i in range(brd.BOARD_SIZE)]
	all_boards = [board]
	p = -1
	for x,y in moves:
		assert board[x][y] == 0
		board[x][y] = p
		all_boards.append(deepcopy(board))
		if shouldPrint:
			brd.printBoard(board)
		winner = brd.decideWinner(board)
		if winner != 0:
			return (winner, all_boards)
		p = -p
	raise ValueError('Winner still not determined after all moves have been made.')


def main():
	data = []
	for filename in argv[1:]:
		print('processing file', filename)
		moves = parseTrainingFile(filename)
		winner, boards = simulate(moves, shouldPrint=True)
		data.extend((b, winner) for b in boards)
	print(data)

if __name__ == '__main__':
	main()


