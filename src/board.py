from sys import stdout

BOARD_SIZE = 20
# The number of pieces that are required to be in a row to win.
COUNT_NEEDED = 5

CHAR_PLAYER_1 = 'X'
CHAR_PLAYER_2 = 'Y'
# Char that represents unclaimed territory.
CHAR_NEUTRAL = '-'
PLAYER_CODES = [CHAR_PLAYER_1, CHAR_NEUTRAL, CHAR_PLAYER_2]


class Board:

	"""
	Access the PLAYER_CODES array to determine which code
	should be used
	"""

	def __init__(self):
		self.board = [[0 for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
		self.next_player = -1

	def _decide_winner_line(self, x, y, dx, dy):
		# Coords at end of vector.
		resting_x = x + COUNT_NEEDED * dx
		resting_y = y + COUNT_NEEDED * dy

		# Check line doesn't leave the board from the left or the top.
		if x < 0 or resting_x < 0:
			return 0
		if y < 0 or resting_y < 0:
			return 0
		# Check line doesn't leave the board from the right or the bottom.
		if x > BOARD_SIZE or resting_x > BOARD_SIZE:
			return 0
		if y > BOARD_SIZE or resting_y > BOARD_SIZE:
			return 0

		start = self.board[x][y]
		for i in range(COUNT_NEEDED):
			if self.board[x][y] != start:
				return 0
			x += dx
			y += dy
		return start

	def print_board(self):
		for row in self.board:
			for coord in row:
				stdout.write(Board.convert_player_char(coord))
			print()
		print()

	def decide_winner(self):
		for y in range(BOARD_SIZE):
			for x in range(BOARD_SIZE):
				for step in [(1, 0), (0, 1), (1, 1), (1, -1)]:
					winner = self._decide_winner_line(x, y, step[0], step[1])
					if winner != 0:
						return winner
		return 0

	@staticmethod
	def convert_player_char(player):
		assert -1 <= player <= 1, "Invalid board cell contents"
		return PLAYER_CODES[player + 1]