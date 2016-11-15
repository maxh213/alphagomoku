from neural_network import conv_network
from sys import argv

#Main file that points to the start of our app
if __name__ == '__main__':
	should_use_save_data = False
	if len(argv) > 1:
		should_use_save_data = argv[1:]
	conv_network(should_use_save_data)



