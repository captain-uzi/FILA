import numpy as np
from arms import *

class sampler(bernoulliArms):
	def __init__(self, arg):
		super().__init__(arg[0])
		self.algo = arg[1]
		self.seed = int(arg[2])
		self.eps = float(arg[3])

	def sample(self):
		'''choose algo'''
		if(self.algo == "epsilon-greedy"):return self.epsilonGreedy()
		if(self.algo == "ucb"):return self.ucb()
		if(self.algo == "kl-ucb"):return self.klUCB()
		if(self.algo == "thompson-sampling"):return self.thompson()
		return self.hintedThompson()


	global argmax, kl, isclose
	def argmax(mat):
		optimal_arms = np.where(mat==np.max(mat))[0]
		argmax = np.random.choice(len(optimal_arms))
		arm = optimal_arms[argmax]
		return arm

	def kl(p,q):
		if(p == 0):
			return (1-p)*np.log((1-p)/(1-q))
		if p==1:
			return p*np.log(p/q)

		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

	def isclose(a, b, precision=1e-06):
		return (abs(a-b) <= precision) 


	def roundRobin(self):
		for arm in range(self.k):
			if(self.armpulls[arm] == 0):
				return self.pull(arm)
		return None

	def epsilonGreedy(self):
		s = np.random.uniform()
		if(s < self.eps):
			arm = np.random.choice(self.k)
		else:
			arm = argmax(self.Pavg)
		return self.pull(arm)
			
	def ucb(self):
		reward = self.roundRobin()
		if(not (reward is None)):
			return reward
		pulls = self.armpulls * 1.0
		uta = np.ones_like(pulls)
		uta[:] *= ( ((2 * np.log(self.totalPulls))) / pulls[:] )**0.5
		ucb = self.Pavg + uta
		arm = argmax(ucb)
		return self.pull(arm)

	def klUCB(self, c = 3, precision = 1e-06):
		reward = self.roundRobin()
		if(not (reward is None)): return reward
		klucb = np.zeros(self.k)
		t = self.totalPulls
		logt_term = np.log(t) + c*np.log(np.log(t))
		for i in range(self.k):
			p = self.Pavg[i]
			RHS = logt_term / self.armpulls[i]
			if(p == 1 or RHS < 0):
				klucb[i] = p
				continue
			lb, ub = p, 1.0
			q = (ub + p)/2.0
			LHS = kl(p,q)
			while(not isclose(LHS , RHS, precision)):
				if(LHS > RHS): ub = q
				elif(LHS < RHS):lb = q
				q = (ub + lb)/2.0
				LHS = kl(p,q)
			klucb[i] = q
		arm = argmax(klucb)
		return self.pull(arm)

	def thompson(self):
		s = self.Psum; 
		f = self.armpulls - s
		beta = np.random.beta(s+1, f+1)
		arm = np.argmax(beta)
		return self.pull(arm)

	def hintedThompson(self):
		hint_ls = self.optimalArm()
		s = self.Psum 
		f = self.armpulls - s
		beta = np.random.beta(s+1, f+1)
		for a in range(self.k):
			beta[a] += np.exp(-self.armpulls[a]*kl(self.Pavg[a], hint_ls))
		arm = np.argmax(beta)
		return self.pull(arm)