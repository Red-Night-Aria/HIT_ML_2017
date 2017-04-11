from numpy import*
import matplotlib.pyplot as plt
X = [[1.00,float(x.strip())] for x in open("q2x.dat").readlines()] 
Y = [float(y.strip()) for y in open("q2y.dat").readlines()]
XY=[]
theta=[0,0]
point = list(zip(X,Y))
matx=mat(X)
strInd=matx[:,1].argsort(0)
xsorted=matx[strInd][ : , 0 , : ]
plt.figure(figsize=(10,10),dpi=150)
plt.title("marisuki_task1_graph")
def solve_matrix(vecx,vecy,tau,specpoint):
	matx=mat(vecx)
	maty=mat(vecy).T
	m=shape(vecx)[0]
	matw=mat(eye(100))
	for i in range(0,100):
		diff= specpoint - matx[i, :]
		matw[i,i]=exp(diff*diff.T/(-2*tau**2))
	XWTX=matx.T*matw*matx
	if linalg.det(XWTX)==0 :
		print('matrix singular')
		return 0
	theta=XWTX.I*(matx.T*(matw*maty))
	return specpoint*theta

def allsolve(vecx,vecy,tau):
	m=shape(vecx)[0]
	YY=zeros(m)
	for i in range(m):
		YY[i]=solve_matrix(vecx,vecy,tau,vecx[i])
	return YY
Test=[0.1,0.3,0.7,1.0,5.0,10.0,50.0]
Color=['r','b','c','k','y','m']
Label=['k=0.1','k=0.3','k=0.5','k=0.7','k=1.0','k=5.0','k=10.0','k=50.0']

for i in range(0,6):
	HTC=allsolve(X,Y,Test[i])
	plt.plot(xsorted[:,1],HTC[strInd],color=Color[i],linestyle='-',label=Label[i])

X = [float(x.strip()) for x in open("q2x.dat").readlines()]
plt.scatter(X,Y,color='g',marker='x',label='t-datas')
plt.legend(loc='upper left')
plt.savefig("marisuki.png")
plt.savefig("marisuki:RuiKang's_task1.pdf")
plt.show()