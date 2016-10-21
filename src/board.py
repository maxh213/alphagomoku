from sys import stdout

from copy import deepcopy
from typing import List
import player

"""
A row consists of a list of players.
"""
RowStruct = List[int]

"""
A board consists of a list of rows.
The board is navigated with [x][y] coordinates.
"""
BoardStruct = List[RowStruct]

BOARD_SIZE = 20
# The number of pieces that are required to be in a row to win.
COUNT_NEEDED = 5


class Board:

	"""
	Access the PLAYER_CODES array to determine which code
	should be used
	"""

	def __init__(self):
		self._board = [[0 for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
		self._possible_moves = deepcopy(self._board)
		self._next_player = -1
		self._winner = 0
		self._winning_moves = None

	def _decide_winner_line(self, x, y, dx, dy):
		# Coords at end of vector.
		resting_x = x + COUNT_NEEDED * dx
		resting_y = y + COUNT_NEEDED * dy

		# Check line doesn't leave the board from the left or the top.
		if x < 0 or resting_x < 0:
			return 0, None
		if y < 0 or resting_y < 0:
			return 0, None
		# Check line doesn't leave the board from the right or the bottom.
		if x > BOARD_SIZE or resting_x > BOARD_SIZE:
			return 0, None
		if y > BOARD_SIZE or resting_y > BOARD_SIZE:
			return 0, None

		start = self._board[x][y]
		moves = []
		for _ in range(COUNT_NEEDED):
			if self._board[x][y] != start:
				return 0, None
			moves.append((x, y))
			x += dx
			y += dy
		return start, moves

	def print_board(self):
		for row in self._board:
			for coord in row:
				stdout.write(player.convert_player_char(coord))
			print()
		print()

	def decide_winner(self):
		return self._winner, self._winning_moves

	def _decide_winner(self, x: int, y: int) -> None:
		"""
		Decides if there is a winner using the provided coords.
		If there is a winner, _winner and _winning_moves are set accordingly.
		:param x: X coord.
		:param y: Y coord.
		"""
		for step in [(1, 0), (0, 1), (1, 1), (1, -1)]:
			winner, moves = self._decide_winner_line(x, y, step[0], step[1])
			if winner != 0:
				self._winner = winner
				self._winning_moves = moves
				return winner, moves
		return 0, None

	def move(self, x: int, y: int, p: int) -> bool:
		"""
		:param x: The x coordinate to move to.
		:param y: The y coordinate to move to.
		:param p: The player.
		:return: True if move successful, False if move invalid.
		"""
		if self._board[x][y] != 0 or not player.is_valid(p) or p != self._next_player or self._winner != 0:
			return False

		self._next_player = -p
		self._board[x][y] = p
		self._decide_winner(x, y)
		return True

	def _remove_move(self, x: int, y: int) -> None:
		"""
		Removes a move from the list of possible moves.
		"""
		self._possible_moves[x][y] = "X"
		# del(self.possibleMoves[x][y])
		print(self._possible_moves)

	def get_board(self) -> BoardStruct:
		"""
		:return: A deep copy of the board 2D array.
		"""
		return deepcopy(self._board)
