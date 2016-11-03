import re

from pip._vendor.distlib.compat import raw_input

from gomoku_app.gomoku import Gomoku
from player import get_player_string


def validate_input(move: str) -> bool:
	pattern = re.compile('([1-9]|1[0-9]|20),([1-9]|1[0-9]|20)')
	return pattern.match(move) is not None


def get_user_move(player: int) -> (int, int):
	move = None
	while True:
		move = raw_input("Type your move " + get_player_string(player) + " (X,Y): ")
		if validate_input(move):
			break
	coordinates = move.split(",")
	x = int(coordinates[0]) - 1
	y = int(coordinates[1]) - 1
	return x, y


def play_game() -> None:
	player = -1
	won = False
	while not won:
		x_coordinate, y_coordinate = get_user_move(player)
		if gomoku.make_move(x_coordinate, y_coordinate, player):
			won = gomoku.check_for_winner()
			player = -player
		else:
			print("Error making move it may be invalid, please check again!")


if __name__ == '__main__':
	gomoku = Gomoku()
	play_game()
