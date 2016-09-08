#! /usr/bin/env python3
import board as brd
from sys import argv


CHAR_PLAYER_1 = 'X'
CHAR_PLAYER_2 = 'Y'
# Char that represents unclaimed territory
CHAR_NEUTRAL = '-'

def convertPlayerChar(player):
	if player == -1:
		return CHAR_PLAYER_1
	if player == 0:
		return CHAR_NEUTRAL
	if player == 1:
		return CHAR_PLAYER_2
	return None

def parseTrainingFile(path):
	with open(path) as f:
		ls = f.readlines()

	ls = ls[1:-3]
	ls = [l.strip() for l in ls]
	ls = [l.split(',')[:2] for l in ls]
	return [(int (m[0]) - 1,int (m[1]) - 1) for m in ls]

def simulateBoard(moves, shouldPrint=False):
	board = [[0 for j in range(brd.BOARD_SIZE)]for i in range(brd.BOARD_SIZE)]
	p = 1
	for x,y in moves:
		if board[x][y] != 0 and shouldPrint:
			print ('Warn: board move invalid')
		board[x][y] = p
		winner = brd.decideWinner(board)
		if winner != 0:
			if shouldPrint:
				print ("Winner found: " + convertPlayerChar(winner))
			return winner
		p = -p
		if shouldPrint:
			brd.printBoard(board)

moves = parseTrainingFile(argv[1])
print(convertPlayerChar(simulateBoard(moves)))

