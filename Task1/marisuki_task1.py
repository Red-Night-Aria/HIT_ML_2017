import numpy as np
import matplotlib.pyplot as plt
X = [float(x.strip()) for x in open("q2x.dat").readlines()] 
Y = [float(y.strip()) for y in open("q2y.dat").readlines()]
theta =np.array([0,0])
step = np.array([1e-4,1e-4])
f = open("theta.dat","w+")
plt.figure(figsize=(10,10),dpi=150)
plt.title("marisuki_task1_graph")
plt.scatter(X,Y)
def h(x):
	return theta[0]*x+theta[1]
def divs(y,x):
	return h(x)-y
def plus(x,y):
	global step
	return [step[0]*divs(y,x)*x, step[1]*divs(y,x)*1]
for j in range(1,1000):
	np.save(f,theta)
	for i in range(1,100):
		theta=theta-plus(X[i],Y[i])
	print(theta)
	if j%20==0 or j<= 10 :
		Xi=np.linspace(-7,13,256,endpoint=True)
		Yi=[theta[0]*xi+theta[1] for xi in Xi]
		plt.plot(Xi,Yi,color="red",linewidth=1.0,linestyle='-')
plt.savefig("marisuki.png")
plt.savefig("marisuki:RuiKang's_task1.pdf")
plt.show()

