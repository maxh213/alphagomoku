import re

from pip._vendor.distlib.compat import raw_input

from gomoku_app.gomoku import Gomoku


def validate_input(move):
	pattern = re.compile('(1[0-9]|[0-9]),(1[0-9]|[0-9])')
	return pattern.match(move) is not None


if __name__ == '__main__':
	gomoku = Gomoku()
	player = -1
	while (1):
		player = -player
		move = None
		while (1):
			move = raw_input("Type your move (X,Y): ")
			if (validate_input(move)):
				break
		coordinates = move.split(",")
		x_coordinate = coordinates[0]
		y_coordinate = coordinates[1]
		gomoku.make_move(int(x_coordinate), int(y_coordinate), player)
