import player
from board import Board


def print_winning_message(winner: int, winning_moves: list) -> None:
	print(player.get_player_string(winner) + " has won using the following moves: ")
	for move in winning_moves:
		print("(" + str(move[0] + 1) + ", " + str(move[1] + 1) + ")")


def is_winner(winner: int) -> bool:
	return winner != 0


class Gomoku:
	board = Board()

	def make_move(self, x: int, y: int, player: int) -> bool:
		if self.board.possible_moves[x][y] != "X":
			self.board.move(x, y, player)
			self.board.remove_move(x, y)
			self.board.print_board()
			return True

	def check_for_winner(self) -> bool:
		winner, winning_moves = self.board.decide_winner()
		if is_winner(winner):
			print_winning_message(winner, winning_moves)
			return True
		return False
