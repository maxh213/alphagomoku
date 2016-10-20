from pip._vendor.distlib.compat import raw_input

from gomoku_app.gomoku import Gomoku

if __name__ == '__main__':
	gomoku = Gomoku()
	player = -1
	while (1):
		player = -player
		move = raw_input("Type your move (X,Y): ")
		coordinates = move.split(",")
		x_coordinate = coordinates[0]
		y_coordinate = coordinates[1]
		gomoku.make_move(int(x_coordinate), int(y_coordinate), player)
