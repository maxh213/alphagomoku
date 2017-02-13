from datetime import datetime, timedelta

from gomokuapp.board import Board, BoardStruct, MoveStruct
from neuralnetwork.neural_network import setup_network, use_network

class Neural_Network:

	def __init__(self):
		self.training_input, self.heuristic, self.keep_prob, self.tf_output, self.sess = setup_network()

	def nn(self, board: Board) -> float:
		return use_network(board.get_board(), self.training_input, self.heuristic, self.keep_prob, self.tf_output, self.sess)


class Node:
	DEFAULT_DEPTH = 20

	"""
	Represents a move that can be made, how good that move is, and what moves can be made after it.
	"""

	def __init__(self, move: MoveStruct, board: Board, neural_network: Neural_Network):
		self.children = []
		self.x, self.y = move
		self._board = board
		self.neural_network = neural_network

		# Value between -1 and 1, where 1 means we've won, and -1 means we've lost.
		self.value = 1 if board.decide_winner() is not None else self.neural_network.nn(board)

	def get_value(self) -> int:
		return self.value

	def get_move(self) -> MoveStruct:
		return self.x, self.y

	def explore(self):
		p = self._board.get_next_player()

		moves = self._board.get_possible_moves()

		#print("Exploring %r,%r: %r" % (self.x, self.y, moves))
		for x, y in moves:
			valid = self._board.move(x, y, p)
			if valid:
				child = Node((x, y), self._board, self.neural_network)
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

	def __init__(self, board: Board, exploration: float = DEFAULT_EXPLORATION, min_time: int = DEFAULT_TIME, max_moves: int = DEFAULT_MAX_MOVES):
		self.board = board
		self.exploration = exploration
		self.calculation_time = timedelta(seconds=min_time)
		self.max_moves = max_moves

		self.states = []
		self.wins = {}
		self.plays = {}
		self.neural_network = Neural_Network()

	def update(self, state: BoardStruct) -> None:
		self.states.append(state)

	def get_play(self):
		begin = datetime.utcnow()
		while datetime.utcnow() - begin < self.calculation_time:
			self.run_simulation()

	def run_simulation(self):
		states_copy = self.states[:] #{:] creates a copy of the list
		state = states_copy[-1] #gets the last element in the states_copy list

		for _ in range(self.max_moves):
			legal = self.board.get_possible_moves()
			x, y = self.choice(legal)
			assert self.board.move(x, y, self.board.get_next_player())
			state = self.board.get_board()
			states_copy.append(state)
			#Change the below to _winner instead of decide_winner cause we want 0 if no winner not none
			winner = self.board._winner
			if winner is not 0: #0 means there is no winner -1, or 1 are the possible winners
				break

	def choice(self, legal_moves):
		'''
			TODO:
			-Add depth
			-Keep history of played moves for future use
			-Use UCB1 to pick branches of the tree when there is enough play data on them.
			-This method
		'''
		print(legal_moves)
		return 1, 1


def main():
	board = Board()
	monte_carlo = MonteCarlo(board)
	monte_carlo.update(BoardStruct)
	monte_carlo.get_play()
