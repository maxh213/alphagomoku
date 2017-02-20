import random
from copy import deepcopy
from datetime import datetime, timedelta

from gomokuapp.board import BOARD_SIZE
from gomokuapp.board import Board, BoardStruct, MoveStruct
from neuralnetwork.neural_network import setup_network, use_network, reset_default_graph


class Neural_Network:
	def __init__(self):
		self.training_input, self.heuristic, self.keep_prob, self.tf_output, self.sess = setup_network()

	def nn(self, board: Board, player) -> float:
		return use_network(board.get_board(), self.training_input, self.heuristic, self.keep_prob, self.tf_output,
		                   self.sess, player)

	'''
		This resets the tensorflow graph and keeps it running at 0.01 seconds per use.

		If this isn't called after at least every 200 calls the time per use for the nn will increase with each call.
	'''

	def clear_garbage_from_nn(self):
		reset_default_graph()
		self.training_input, self.heuristic, self.keep_prob, self.tf_output, self.sess = setup_network()


class Node:
	DEFAULT_DEPTH = 3
	DEFAULT_BREADTH = 2

	"""
	Represents a move that can be made, how good that move is, and what moves can be made after it.
	"""

	def __init__(self, move: MoveStruct, board: Board, neural_network: Neural_Network, player_for_computer: int):
		self.children = []
		self.x, self.y = move
		self._board = board
		self.neural_network = neural_network
		self.debug_nn_outputs = []
		self.player = self._board.get_next_player() * -1
		self.player_for_computer = player_for_computer

		# Value between -1 and 1, where 1 means we've won, and -1 means we've lost.
		self.value = 10 if board.decide_winner()[0] is not 0 else self.neural_network.nn(board, self.player)

	def get_value(self) -> int:
		return self.value

	def get_move(self) -> MoveStruct:
		return self.x, self.y

	def explore(self):
		player = self._board.get_next_player()
		played_moves = self._board.get_played_moves()
		if len(played_moves) > 0:
			moves = self.get_adjacent_moves(played_moves)
		else:
			# Computer goes first so try from 10 random moves. Will most likely want to change at some point
			moves = self.pick_from_random()
		# moves = self._board.get_possible_moves()
		# print("Exploring %r,%r: %r" % (self.x, self.y, moves))
		for x, y in moves:
			# print(x,y,self.player)
			valid = self._board.move(x, y, player)
			if valid:
				next_board = deepcopy(self._board)
				child = Node((x, y), next_board, self.neural_network, self.player_for_computer)
				self.debug_nn_outputs.append({x, y, child.get_value()})
				self.children.append(child)
				reversed_move = self._board.reverse_move()
				assert reversed_move == (x, y, player), "%r vs %r" % (reversed_move, (x, y, player))
		print(self.debug_nn_outputs)

		self.children = sorted(self.children, key=lambda child: child.get_value(), reverse=True)

	def select(self, depth: int = DEFAULT_DEPTH) -> "Node":
		if len(self.children) == 0:
			self.explore()
			if self.player != self.player_for_computer:
				self.value = 1 - self.value
			depth -= 1

		self.children = sorted(self.children, key=lambda child: child.get_value(), reverse=True)
		children_to_explore = self.children[:self.DEFAULT_BREADTH]

		if depth > 0:
			for child in children_to_explore:
				child.select(depth)

		for child in self.children:
			self.value += child.get_value()

		children_to_explore = sorted(children_to_explore, key=lambda child: child.get_value(), reverse=True)

		return children_to_explore[0] if children_to_explore else None

	def get_adjacent_moves(self, played_moves: list) -> list:
		adjacent_moves = []
		for move in played_moves:
			x = move[0]
			y = move[1]
			go_up = (x, y - 1)
			go_up_right = (x - 1, y - 1)
			go_right = (x - 1, y)
			go_down_right = (x - 1, y + 1)
			go_down = (x, y + 1)
			go_down_left = (x + 1, y + 1)
			go_left = (x + 1, y)
			go_up_left = (x + 1, y - 1)
			adjacent_moves.extend(
				(go_up, go_up_right, go_right, go_down_right, go_down, go_down_left, go_left, go_up_left))
		adjacent_moves = list(
			filter(lambda move: move not in played_moves and self.valid_coordinate(move), adjacent_moves))
		return adjacent_moves

	def valid_coordinate(self, move: MoveStruct) -> bool:
		x = move[0]
		y = move[1]
		return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

	def pick_from_random(self) -> list:
		random_moves = []
		for i in range(0, 10):
			x = random.randint(0, BOARD_SIZE - 1)
			y = random.randint(0, BOARD_SIZE - 1)
			random_moves.append((x, y))
		return random_moves


class MonteCarlo:
	"""
	Built on work by Jeff Bradberry: https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search
	"""

	DEFAULT_EXPLORATION = 2

	DEFAULT_TIME = 10

	DEFAULT_MAX_MOVES = 100

	def __init__(self, board: Board, exploration: float = DEFAULT_EXPLORATION, min_time: int = DEFAULT_TIME,
	             max_moves: int = DEFAULT_MAX_MOVES):
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
		states_copy = self.states[:]  # [:] creates a copy of the list
		state = states_copy[-1]  # gets the last element in the states_copy list

		for _ in range(self.max_moves):
			legal = self.board.get_possible_moves()
			x, y = self.choice(legal)
			assert self.board.move(x, y, self.board.get_next_player())
			state = self.board.get_board()
			states_copy.append(state)
			winner = self.decide_winner()
			if winner is not 0:  # 0 means there is no winner -1, or 1 are the possible winners
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
