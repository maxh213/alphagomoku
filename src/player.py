"""
The player is represented by an int.
-1:	Player 1
0:	Not occupied.
1:	Player 2
"""

CHAR_PLAYER_1 = 'X'
CHAR_PLAYER_2 = 'Y'
# Char that represents unclaimed territory.
CHAR_NEUTRAL = '-'
PLAYER_CODES = [CHAR_PLAYER_1, CHAR_NEUTRAL, CHAR_PLAYER_2]


def is_valid(player: int) -> bool:
	"""
	:param player: integer representing the player.
	:return: True if int between -1 and 1, False otherwise.
	"""
	return -1 <= player <= 1

def convert_player_char(player: int) -> str:
	assert -1 <= player <= 1, "Invalid board cell contents"
	return PLAYER_CODES[player + 1]