from board import Board, BoardStruct, MoveStruct
from datetime import datetime, timedelta

from neural_network import use_network

class _Network:
	"""
	Todo: turn NN into a class so that this is unnecessary.
	Holds the state of the use_network function.
	"""

	def __init__(self):
		self.first_use = True

	def nn(self, board:Board) -> float:
		last_move = board.get_last_move()
		if last_move is None:
			return 0
		x, y = board.get_last_move()
		# Mock nn that favours moves further to the right of the board.
		v = (x - 10) / 20 + (y-10) / 20
		print("%d,%d= %f" % (x, y, v))
		return v


class Node:

	DEFAULT_DEPTH = 20

	"""
	Represents a move that can be made, how good that move is, and what moves can be made after it.
	"""

	def __init__(self, move: MoveStruct, board: Board, net: _Network=None):
		self.children = []
		self.x, self.y = move
		self._board = board
		if net is None:
			net = _Network()

		# Value between -1 and 1, where 1 means we've won, and -1 means we've lost.
		self.value = 1 if board.decide_winner() is not None else net.nn(board)

	def get_value(self) -> int:
		return self.value

	def get_move(self) -> MoveStruct:
		return self.x, self.y

	def explore(self):
		p = self._board.get_next_player()

		moves = self._board.get_possible_moves()

		print("Exploring %r,%r: %r" % (self.x, self.y, moves))
		for x, y in moves:
			valid = self._board.move(x, y, p)
			assert valid
			child = Node((x, y), self._board)
			self.children.append(child)
			reversed_move = self._board.reverse_move()
			assert reversed_move == (x, y, p), "%r vs %r" % (reversed_move, (x, y, p))

		self.children = sorted(self.children, key=lambda child: child.get_value(), reverse=True)

	def select(self, depth: int = DEFAULT_DEPTH) -> "Node":
		if len(self.children) == 0:
			self.explore()

		return self.children[0]



class MonteCarlo:
	"""
	Built on work by Jeff Bradberry: https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search
	"""

	DEFAULT_EXPLORATION = 2

	DEFAULT_TIME = 10

	DEFAULT_MAX_MOVES = 100

	def __init__(self, board: Board, exploration: float=DEFAULT_EXPLORATION, min_time: int=DEFAULT_TIME,
				 max_moves: int=DEFAULT_MAX_MOVES):
		self.board = board
		self.exploration = exploration
		self.calculation_time = timedelta(seconds=min_time)
		self.max_moves = max_moves

		self.states = []
		self.wins = {}
		self.plays = {}

	def update(self, state: BoardStruct) -> None:
		self.states.append(state)

	def get_play(self):
		begin = datetime.utcnow()
		while datetime.utcnow() - begin < self.calculation_time:
			self.run_simulation()

	def run_simulation(self):
		states_copy = self.states[:]
		state = states_copy[-1]

		for _ in range(self.max_moves):
			legal = self.board.get_possible_moves()
			x, y = self.choice(legal)
			assert self.board.move(x, y, self.board.get_next_player())
			state = self.board.get_board()
			states_copy.append(state)

			winner, _ = self.board.decide_winner()

			if winner is not 0:
				break

	def choice(self):
		pass