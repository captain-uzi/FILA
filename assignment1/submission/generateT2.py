from bandit import bandit
import sys
import time 
import math
import matplotlib.pyplot as plt
methods = ['thompson-sampling', 'thompson-sampling-with-hint']
epsilon = 0.02 #0.999, 0.001
horizons = [100, 400, 1600, 6400, 25600, 102400] #[102400]
instances = ["../instances/i-1.txt","../instances/i-2.txt", "../instances/i-3.txt",]

def generate_graph(todo):
	start = time.time()
	inst = 0
	for i in instances:
		print(i)
		x = {}
		y = {}
		for algo in todo:
			x[algo]=[]
			y[algo]=[]
			f = open("outputDataT2.txt","a")
			#f = open("../data-for-graph/"+"i-"+i[-5]+"/"+str(algo)+".txt","a")
			#g = open("../data-for-graph/"+str(algo)+".txt","a")
			#g.write(i+"\n")
			#print(algo)
			for hz in horizons:
				regret = 0.0
				for seed in range(50):
					args = [i, algo, seed, epsilon, hz]
					bandit_instance = bandit(args)
					REG = bandit_instance.run()
					regret += REG
					#write file
					f.write(i+', '+algo+', '+str(seed)+', '+str(epsilon)+', '+str(hz)+', '+str(REG)+'\n')
					#print progress
					# sys.stdout.write("\rseed: %i, time elapsed %.2f" % (seed ,(time.time()-start)))
					# sys.stdout.flush()
				regret /= 50.0
				x[algo].append(math.log(hz))
				y[algo].append(regret)
				#print("\nhorizon:", hz, "Regret:", regret)
				#g.write(i+"  ----"+str(regret)+"\n")
			f.close()
		plt.title("Task 2 Comparision for: "+i)
		plt.plot(x[todo[0]],y[todo[0]], label='thompson-sampling')
		plt.plot(x[todo[1]],y[todo[1]], label='thompson-sampling-with-hint')
		plt.legend(loc='upper left', frameon=True)
		plt.ylabel('Regret')
		plt.xlabel('Horizon (log scale)')
		plt.show()
			#g.close()
	print("time taken:",time.time()-start)

if __name__ == '__main__':
	print(instances, horizons)
	generate_graph(methods) #the final boss