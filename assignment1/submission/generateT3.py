from bandit import bandit
import sys
import time 

method = 'epsilon-greedy'
epsilons = [0.02, 0.999, 0.001]
horizon = 102400
instances = ["../instances/i-1.txt","../instances/i-2.txt", "../instances/i-3.txt",]

def generate_graph(todo):
	start = time.time()
	#jeeeezzz
	inst = 0
	for i in instances:
		print(i)
		print(todo)
		for epsilon in epsilons:
			regret = 0.0
			for seed in range(50):
				args = [i, todo, seed, epsilon, horizon]
				bandit_instance = bandit(args)
				REG = bandit_instance.run()
				regret += REG
				sys.stdout.write("\rseed: %i, time elapsed %.2f" % (seed ,(time.time()-start)))
				sys.stdout.flush()
			regret /= 50.0
			print("\nepsilon:", epsilon, "Regret:", regret)
	print("time taken:",time.time()-start)

if __name__ == '__main__':
	print(instances, horizon)
	generate_graph(method) 
	