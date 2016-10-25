from board import Board


class Gomoku:
	board = Board()

	def make_move(self, x, y, player):
		if self.board._possible_moves[x][y] != "X":
			self.board._board[y][x] = player
			self.board._remove_move(x, y)
			self.board.print_board()
			return True
