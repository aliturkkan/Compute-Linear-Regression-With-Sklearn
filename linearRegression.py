from matplotlib.style import use
from numpy import loadtxt, size, linspace
from sklearn.linear_model import LinearRegression as LR
from matplotlib.pyplot import scatter, show, legend

#load dataset
data = loadtxt('deneme.txt',delimiter=',')

#dataset's first col = x component
#dataset's second col = y component
x = data[:,0]
y = data[:,1]

#the length of the x component of the data set
s = size(x)

#Requirements for calculating linear regression
x = x.reshape(s,1)
y = y.reshape(s,1)

#linear line drawing and optimal placement
lineerR = LR()
lineerR.fit(x,y)
lineerR.predict(x)

#slope calculation
slope = float(lineerR.coef_)
#The intersection point of y for x = 0
interSec = float(lineerR.intercept_)

print("The slope of the guess line plotted against the input data:\t{0}".format(slope))
print("The intersection point of y for x = 0:\t{0}".format(interSec))

#print equation of prediction line
if(interSec<0):
    print("equation of prediction line:\ty = {0}x{1}".format(round(slope,4),round(interSec,4)))
elif(interSec>=0):
    print("equation of prediction line:\ty = {0}x+{1}".format(round(slope, 4), round(interSec, 4)))
else:
    print("equation of prediction line:\ty = {0}x".format(round(slope, 4)))

#finding the lowest and highest values to show the data on the plot
j = sorted(x)
#etc. linspace(lowest, highest, frequency)
theta = linspace(j[0], j[s-1], j[0]*j[s-1]*2)

#plotting the data and linear regression line
use('ggplot')
scatter(x,y,color="r",label="input data")
scatter(theta, theta * slope + interSec, linewidths=1, color="b", label="Linear Regression Line")
legend()
show()