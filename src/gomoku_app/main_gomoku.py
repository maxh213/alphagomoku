from gomoku_app.gomoku import Gomoku

if __name__ == '__main__':
	gomoku = Gomoku()
	gomoku.make_move(19, 19, -1)
	gomoku.make_move(19, 18, 1)
