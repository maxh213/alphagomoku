from board import Board


class Gomoku:
	board = Board()

	def make_move(self, x, y, player):
		self.board._board[y][x] = player
		self.board.print_board()
