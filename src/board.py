from copy import deepcopy
from sys import stdout
from typing import List, Tuple

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

"""
Moves are represented as a tuple of x and y coords.
"""
MoveStruct = Tuple[int, int]

MovesStruct = List[MoveStruct]

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
		self.possible_moves = deepcopy(self._board)
		self._next_player = -1
		self._winner = 0
		self._winning_moves = None

	def print_board(self):
		coords = range(BOARD_SIZE)
		for y in reversed(coords):
			for x in coords:
				coord = self._board[x][y]
				stdout.write(player.convert_player_char(coord))
			print()
		print()

	def decide_winner(self):
		return self._winner, self._winning_moves

	def _decide_winner_line(self, x: int, y: int, dx: int, dy: int) -> Tuple[int, MovesStruct]:
		"""
		Counts the number of spaces in a line belonging to the player in the given space.
		So if board[x][y] belongs to player 1, dx = 1, and dy = 0,
		then this function will search the horizontal line for consecutive player 1 spaces from board[x][y].
		"""
		p = self._board[x][y]
		count = 1
		moves = [(x, y)]
		for d in [-1, 1]:
			tx = x + dx * d
			ty = y + dy * d
			while check_coords(tx, ty) and self._board[tx][ty] == p:
				moves.append((tx, ty))
				count += 1
				if count == COUNT_NEEDED:
					return p, moves
				tx += dx * d
				ty += dy * d
		return 0, None

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

	def remove_move(self, x: int, y: int) -> None:
		"""
		Removes a move from the list of possible moves.
		"""
		self.possible_moves[x][y] = "X"

	# del(self.possibleMoves[x][y])
	# print(self._possible_moves)

	def get_board(self) -> BoardStruct:
		"""
		:return: A deep copy of the board 2D array.
		"""
		return deepcopy(self._board)


def check_coords(x: int, y: int) -> bool:
	return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE
