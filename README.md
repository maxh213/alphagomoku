# Alpha Gomoku Group Project
#### Written and maintained by Matthew Boakes, Harry Clarke, Matthew Clayton, Max Harris, and Jamie Pont.

### How to install (Linux)
* clone the directory from git
* install python3
* pip install tensorflow
* run sudo sh ./setup.sh

### How to install (Windows)
* Same as Linux but instead of running setup.sh download the training data
 * https://drive.google.com/uc?export=download&id=0B86_99L1GbtLY19tTlllekg1N0k
* In the root directory unzip the save_data folder into resources/training

### How to run
* To train the neural network:
 * python3 neural_network_main.py
  * To run it with previously trained weights pass in the following parameter to the above command: true
 * To play gomoku against the trained neural network and tree search:
  * python3 gomoku_main.py

### Introduction

The AlphaGomoku project aims to bring the concept behind Deep Mind's AlphaGo ("Mastering the game of Go with deep neural networks and tree search", David Silver et al., 28/01/16) to a game called "Gomoku".

### Project Goals
* September
 * Finish learning from the book
* October 
 * Have a basic neural netowkr put together for our understanding
 * Finish Gomoku App for playing the game
* End of Christmas 
 * Apply knowledge gained from basic NN project and develop a NN around gomoku
* After Christmas
 * Fine tune/finish project

