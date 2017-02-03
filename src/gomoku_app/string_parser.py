import re
from board import Board


"""
	"Brain" refers to the Neural Network/application.
	"Manager" refers to the third-party brokering the games.
"""

class string_parser:

	board = None
	while(True):
		def read(input: str) -> (str):
			if(input == "START 20"):
				board = Board()
			"""
			When it receives this command (99% of the time it will be 20, but not always),
			the brain will create a new board of whatever size is defined in the command,
			then respond "OK".
			If the brain doesn't like the size, respond with "ERROR". There can be a message
			after that. The manager can try other sizes, or it can display a message to the
			user. Respond OK if the board is initialized successfully.
			The brain will respond with "OK" if the board initiatites correctly.
			The brain will respond with "ERROR" (and an optional message) if it does not.
			"""
			if (input == "BEGIN"):
				"""
			After receiving a START command and initiating the board, "BEGIN" states that the
			brain gets the first move.
			The brain will respond with the X and Y coordinates of its move.
			"""
			if(input == "TURN X,Y"):
				"""
			X and Y represent the X and Y coordinates of the opponent's move.
			The brain will respond with the X and Y coordinates of its move.
			"""
			if(input == "BOARD"):
				"""
			"BOARD" creates a new match, but with the potential to set the board to however the
			manager wishes it to be. After receiving the command, new lines will be of format
			"X,Y,[Player (1/2)]", until it sends "DONE".
			The brain will then answer with the coordinates of its next move.
			Manager:
				BOARD
				10,10,1
				10,11,2
				9,10,1
				DONE
			Brain:
				9,9
			(This assumes the brain is player 2 I assume)
			PS: Moves aren't necessarily sent in the order they were played in, unless using Renju rules.
			"""
			if(input == "INFO [key] [value]"):
				"""
			"INFO" sends the brain some information about the game so everyone knows the rules.
			For instance, "INTO timeout_match 300000" sets the time limit for the whole match
			to be 300,000 milliseconds, or about 5 minutes.
			Full key of INFO commands:
			timeout_turn  - time limit for each move (milliseconds, 0=play as fast as possible)
			timeout_match - time limit of a whole match (milliseconds, 0=no limit)
			max_memory    - memory limit (bytes, 0=no limit)
			time_left     - remaining time limit of a whole match (milliseconds)
			game_type     - 0=opponent is human, 1=opponent is brain, 2=tournament, 3=network tournament
			rule          - bitmask or sum of 1=exactly five in a row win, 2=continuous game, 4=renju
			evaluate      - coordinates X,Y representing current position of the mouse cursor
			folder        - folder for persistent files [Used to determine a folder for persistent data.
							Because this folder is common for all the brains, the brain must create
							its own subfolder with a name exactly matching the brain's name. If the manager
							never sends a folder command, then the brain cannot save permenant files.]
			Manager:
				INFO timeout_milliseconds 300000
			Brain:
				<no answer expected, just store the value somewhere>
			"""
			if(input == "END"):
				"""
			The brain is not expected to answer to this instruction, but rather to delete all of its temporary files.
			"""
			if(input == "ABOUT"):
				"""
			In the format key="value", the brain replies with some details about itself.
			Manager:
				ABOUT
			Brain:
				name="GroupProjectNN", version="1.0", author="#Banterbury", country="UK"
			"""