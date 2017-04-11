import numpy as np
import matplotlib.pyplot as plt

def Lwr_Single(x, tau):
	X_mat = np.mat(X)
	Y_mat = np.mat(Y)
	W = np.diag([np.exp((x-t)**2 / (-2*tau**2)) for t in X])
	return np.linalg.det((X_mat * W * X_mat.T).I * Y_mat * W * X_mat.T)

def Lwr(idx, tau):
	theta = np.array([Lwr_Single(x, tau) for x in X])

	Py = np.array(list(map(lambda x, y: x*y, theta, X)))
	Pt = np.array(sorted(list(map(lambda x, y: (x, y), X, Py))))

	plt.subplot(420+idx+1)
	plt.scatter(X, Y, c='r', s=1, marker='o')
	plt.plot(Pt[:,0], Pt[:,1], "-", linewidth=1, label='τ='+str(tau))
	plt.legend()

X = list([float(x.strip()) for x in open("q2x.dat").readlines()])
Y = list([float(x.strip()) for x in open("q2y.dat").readlines()])

fig = plt.figure(figsize=(20,10))
fig.suptitle("MMMMMua's Graphics", fontsize=25)

Tau = np.logspace(-2, 1, 8)
for idx, tau in enumerate(Tau):
	Lwr(idx, tau)

plt.show()
plt.savefig("MMMMMua.png", dpi=300)
