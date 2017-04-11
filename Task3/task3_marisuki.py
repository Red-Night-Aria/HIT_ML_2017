#import:
from numpy import*
import matplotlib.pyplot as plt
from matplotlib import *
#read:
XY =[]
X=[]
for i in open("task3_x.dat").readlines():
	x,y=map(float,i.strip().split(' '))
	XY.append((x,y))
	X.append((x**3,y**3,x**2*y,x*y**2,x**2,y**2,x*y,x,y,1))
X=mat(X)
XY=mat(XY)
Classify = mat([float(y.strip()) for y in open("task3_y.dat").readlines()]).T

#init:
step=1e-3


#function:
def phy(z):
	return 1.0/(1.0+exp(-z))

def NetT(Arguments,Classification):
	x,y=shape(Arguments)
	theta=ones((y,1))
	for i in range(1,5000):
		hold=theta[0]
		pre=phy(Arguments*theta)
		diff=Classification-pre
		theta=theta+step*Arguments.T*diff
		print(theta)
		print(theta[0]-hold)
	return theta


#dataset:
Parametre=NetT(X,Classify)

#print map:
 #point:
TX=[]
FX=[]
TY=[]
FY=[]
for i in range(0,250):
	if Classify[i]>=0.5:
		TX.append(XY[i,0])
		TY.append(XY[i,1])
	else :
		FX.append(XY[i,0])
		FY.append(XY[i,1])

plt.scatter(TX,TY,color='r',marker='x')
plt.scatter(FX,FY,color='g',marker='o')

 #func:
x=arange(-5,10,0.1)
y=arange(-5,5,0.1)
x,y=meshgrid(x,y)
func=Parametre[0,0]*x**3+Parametre[1,0]*y**3+Parametre[2,0]*x**2*y+Parametre[3,0]*x*y**2+Parametre[4,0]*x**2+Parametre[5,0]*y**2+Parametre[6,0]*x*y+Parametre[7,0]*x+Parametre[8,0]*y+Parametre[9,0]
plt.contour(x,y,func,0)
plt.savefig("marisuki.png")
plt.savefig("marisuki:RuiKang's_task1.pdf")
plt.show()
