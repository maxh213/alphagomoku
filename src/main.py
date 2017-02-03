from typing import Tuple

from neural_network import neural_network_train
from sys import argv

ShouldSaveStruct = bool
ArgsStruct = Tuple[ShouldSaveStruct]


def parse_args(args: str) -> ArgsStruct:
	'''
	Responsible for parsing args into the form of the ArgsStruct.
	'''
	should_use_save = False
	for arg in args:
		if arg.lower() == 'true':
			should_use_save = True
	return should_use_save


def main():
	if __name__ == '__main__':
		should_use_save_data = parse_args(argv[1:])
		neural_network_train(should_use_save_data)

if __name__ == "__main__":
	main()