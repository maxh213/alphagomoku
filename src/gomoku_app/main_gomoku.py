import re

from pip._vendor.distlib.compat import raw_input

from gomoku_app.gomoku import Gomoku


def validate_input(move):
	pattern = re.compile('([1-9]|1[0-9]|20),([1-9]|1[0-9]|20)')
	return pattern.match(move) is not None


if __name__ == '__main__':
	gomoku = Gomoku()
	player = -1
	while (1):
		move = None
		while (1):
			move = raw_input("Type your move (X,Y): ")
			if (validate_input(move)):
				break
		coordinates = move.split(",")
		x_coordinate = int(coordinates[0])-1
		y_coordinate = int(coordinates[1])-1
		if(gomoku.make_move(x_coordinate, y_coordinate, player)):
			player = -player
