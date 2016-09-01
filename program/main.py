#! /usr/bin/env python3
from sys import argv

# The size of the board.
BOARD_SIZE = 20
# The number of pieces that are required to be in a row.
COUNT_NEEDED = 5

# Char that represents player 1
CHAR_PLAYER_1 = 'X'
# Char that represents player 2
CHAR_PLAYER_2 = 'Y'
# Char that represents unclaimed territory
CHAR_NEUTRAL = '-'

def decideWinnerLine (board, x, y, dx, dy):
	if x >= BOARD_SIZE - COUNT_NEEDED * dx:
		return 0
	if y >= BOARD_SIZE - COUNT_NEEDED * dy:
		return 0

	start = board[x][y]
	for i in range(COUNT_NEEDED):
		if board[x][y] != start:
			return 0
		x+=dx
		y+=dy
	return start

def decideWinner (board):
	for y in range(BOARD_SIZE):
		for x in range(BOARD_SIZE):
			for step in [(1,0),(0,1),(1,1)]:
				winner = decideWinnerLine(board, x, y, step[0], step[1])
				if winner != 0:
					return winner
	return 0

def convertPlayerChar(player):
	if player == -1:
		return CHAR_PLAYER_1
	if player == 0:
		return CHAR_NEUTRAL
	if player == 1:
		return CHAR_PLAYER_2
	return None

def printBoard (board):
	for row in board:
		for coord in row:
			print(convertPlayerChar(coord)+ ".", end="")
		print()
	print ()


with open ( argv[1] ) as f:
	ls = f.readlines()

ls = ls[1:-3]
ls = [l.strip() for l in ls]
ls = [l.split(',')[:2] for l in ls]
moves = [(int (m[0]) - 1,int (m[1]) - 1) for m in ls]
print (moves)

board = [[0 for j in range(BOARD_SIZE)]for i in range(BOARD_SIZE)]
p = 1
for x,y in moves:
	if board[x][y] != 0:
		print ('Warn: board move invalid')
	board[x][y] = p
	winner = decideWinner(board)
	if winner != 0:
		print ("Winner found: " + convertPlayerChar(winner))
		break
	p = -p
	printBoard(board)
