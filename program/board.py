BOARD_SIZE = 20
# The number of pieces that are required to be in a row to win.
COUNT_NEEDED = 5

# Char that represents player 1
CHAR_PLAYER_1 = 'X'
# Char that represents player 2
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


def printBoard (board):
	for row in board:
		for coord in row:
			print(convertPlayerChar(coord), end="")#+ ".", end="")
		print()
	print ()

def decideWinnerLine (board, x, y, dx, dy):
	assert dx >= 0
	if x >= BOARD_SIZE - COUNT_NEEDED * dx:
		return 0
	if y >= BOARD_SIZE - COUNT_NEEDED * dy:
		return 0
	if y < COUNT_NEEDED * dy:
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
			for step in [(1,0),(0,1),(1,1),(1,-1)]:
				winner = decideWinnerLine(board, x, y, step[0], step[1])
				if winner != 0:
					return winner
	return 0

