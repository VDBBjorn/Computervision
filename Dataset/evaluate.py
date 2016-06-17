from numpy import *
#import matplotlib.pyplot as plt

seq="01"

gt=genfromtxt("%s/gtdistances.txt"%seq)[:,1]/3.6
res=genfromtxt("%s/results.txt"%seq)[:,1]/3.6
speeds=genfromtxt("%s/actualspeeds.txt"%seq)[:,1]

distance=0
time=0
for i in range(len(gt)):
	if res[i]>gt[i]:
		print("Sorry, you crashed! Speed in frame %d exceeded safety limits."%i)
		break
	else:
		distance+=speeds[i]
		time+=speeds[i]/res[i]
	if time>=60:
		print("You ran out of time after %d frames!"%i)
		break
print("You travelled %f metres in %f seconds."%(distance,time))

#plt.figure()
#plt.plot(gt)
#plt.plot(speeds)
#plt.ylim([0,30])
#plt.show()