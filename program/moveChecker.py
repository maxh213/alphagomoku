import re #regex 

def getFileLength(fileName):
	with open(fileName) as file:
		for i, line in enumerate(file): 
			pass #pass means do nothing
	return i

def getBoardSize(lineOne):
	#lineOne may look like this: Piskvorky 20x20, 11:11, 0
	boardSizeString = lineOne.split(',')[0]
	#boardSizeString may look like this: Piskvorky 20x20
	pattern = re.compile(r'\d+x\d+')
	boardSize = pattern.findall(boardSizeString)
	#boardSize may look like this: ['20x20']
	return int(boardSize[0][:2]) #:2 gets the first two letters of a string

def getMoveHistory(moves):
	moveHistory = [] #you can't reassign variables in a for loop in python..
	for move in moves:
		move = move.split(',')
		move = [int(move[0]), int(move[1])]
		moveHistory = moveHistory + [move]
	return moveHistory	


#Basically if the move is inside the board and hasn't been places before the move is legal
def calculateMoveLegal(content, lineNumber):
	boardSize = getBoardSize(content.pop(0))
	moveHistory = getMoveHistory(content)
	moveInQuestion = moveHistory[lineNumber-1]
	if moveInQuestion[0] > boardSize or moveInQuestion[1] > boardSize:
		return False
	if moveHistory.count(moveInQuestion) > 1:
		return False
	return True


#Main method
def isMoveLegal(fileName, lineNumber):
	assert lineNumber <= getFileLength(fileName), "Line number too big"
	assert lineNumber > -1, "Line number too small"
	with open(fileName) as file:
		#load file into memory
		content = file.read().splitlines() 
	return calculateMoveLegal(content, lineNumber)

def isLatestMoveLegal(fileName):
	return isMoveLegal(fileName, getFileLength(fileName))

#Uncomment for testing
#print(isLatestMoveLegal("unfinishedDevTestGame.psq"))
