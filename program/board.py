BOARD_SIZE = 20
# The number of pieces that are required to be in a row to win.
COUNT_NEEDED = 5

def printBoard (board):
	for row in board:
		for coord in row:
			print(convertPlayerChar(coord)+ ".", end="")
		print()
	print ()

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

