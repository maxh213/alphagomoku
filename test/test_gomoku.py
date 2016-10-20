import unittest

from gomoku_app.gomoku import Gomoku


class TestGomokuApp(unittest.TestCase):
	def setUp(self):
		self.gomoku = Gomoku()


if __name__ == '__main__':
	unittest.main()
