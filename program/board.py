BOARD_SIZE = 20
# The number of pieces that are required to be in a row to win.
COUNT_NEEDED = 5

CHAR_PLAYER_1 = 'X'
CHAR_PLAYER_2 = 'Y'
# Char that represents unclaimed territory
CHAR_NEUTRAL = '-'
PLAYER_CODES = [CHAR_PLAYER_1, CHAR_NEUTRAL, CHAR_PLAYER_2]


"""Access the PLAYER_CODES array to determine which code
should be used"""
def convertPlayerChar(player):
	assert -1 <= player <= 1, "Invalid board cell contents"
	return PLAYER_CODES[player + 1]

def printBoard (board):
	for row in board:
		for coord in row:
			print(convertPlayerChar(coord), end="")#+ ".", end="")
		print()
	print ()

def decideWinnerLine (board, x, y, dx, dy):
	# Coords at end of vector.
	restingX = x + COUNT_NEEDED * dx
	restingY = y + COUNT_NEEDED * dy

	# Check line doesn't leave the board from the left or the top.
	if x < 0 or restingX < 0:
		return 0
	if y < 0 or restingY < 0:
		return 0
	# Check line doesn't leave the board from the right or the bottom.
	if x > BOARD_SIZE or restingX > BOARD_SIZE:
		return 0
	if y > BOARD_SIZE or restingY > BOARD_SIZE:
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

