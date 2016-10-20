from board import Board


class Gomoku:
	board = Board()

	def make_move(self, x, y, player):
		self.board.board[y][x] = player
		self.board.print_board()
