from neural_network import neural_network_train
from sys import argv

#Main file that points to the start of our app
if __name__ == '__main__':
	should_use_save_data = False
	if len(argv) > 1:
		should_use_save_data = argv[1:]
	neural_network_train(should_use_save_data)



