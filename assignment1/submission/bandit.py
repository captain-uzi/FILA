#usr/bin/env/python3
import sys
from algorithms import sampler
import numpy as np

class bandit(sampler):
	def __init__(self, arg):
		super().__init__(arg[:-1])
		self.hz = int(arg[4])

	def run(self, seeded = True):
		REW = 0.0
		if(seeded): np.random.seed(self.seed)
		for i in range(self.hz):
			rewards = self.sample()
			REW += np.sum(rewards)
		REG = float(self.hz)*self.optimalArm() - REW
		return REG

def main():
	if(len(sys.argv) != 11):
		print("Please enter valid arguments")
		sys.exit()
	bandit_instance = bandit(sys.argv[2::2])
	for arg in sys.argv[2::2]:
		print(arg, end = ", ")

	print(bandit_instance.run())

if __name__ == '__main__':
	main()

