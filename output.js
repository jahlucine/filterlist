import statistics
import numpy as np
# Import the necessary packages and modules
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from numpy import diff
from sklearn.preprocessing import StandardScaler

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def differ(x,n):
    res=[]
    for i in range(0,n):
        res.append(0)
    for i in range(n,len(x)):
        res.append(x[i]-x[i-n])
    return np.array(res)

def classer(x):
    res=[]
    for i in x:
        if i>=0.5:
            res.append(1)
        else:
            if i<=-0.5:
                res.append(-1)
            else:
                res.append(0)
    return np.array(res)
                
with open("data-training.csv", "r") as ins:
    array = []
    deb=0
    lastval=0
    buff=[0] * 30
    rightsum=0
    errsum=0
    targ=[]
    show=0
    ask=[]
    bid=[]
    mid=[]
    spread=[]
    diffs=[]
    nasks=[]
    nbids=[]
    volsum=[]
    voldif=[]
    ndir=[]
    for line in ins:
        if deb==0:
            #print(line)
            arr = line.split(",")
            #print(arr[30:45])
        if deb>0:
            arr = line.split(",")
            askrate=arr[0:15]
            askrate=[x for x in askrate if x]
            askrate = [float(i) for i in askrate]
            nasks.append(len(askrate))
            askmin=min(askrate)/100
            bidrate=arr[30:45]
            bidrate=[x for x in bidrate if x]
            bidrate = [float(i) for i in bidrate]
            nbids.append(len(bidrate))
            ndir.append(len(askrate)-len(bidrate))
            bidmax=max(bidrate)/100
            askvols=arr[15:30]
            askvols=[x for x in askvols if x]
            askvols = [float(i) for i in askvols]
            askvol=sum(askvols)
            bidvols=arr[45:60]
            bidvols=[x for x in bidvols if x]
            bidvols = [float(i) for i in bidvols]
            bidvol=sum(bidvols)
            volsum.append(askvol+bidvol)
            voldif.append(askvol-bidvol)
            yy=float(arr[60])
            thismean=(askmin+bidmax)/2
            diff=thismean-lastval
            diffs.append(diff)
            targ.append(yy)
            ask.append(askmin)
            bid.append(bidmax)
            mid.append(thismean)
            spread.append(diff)
            if deb%50000==0:
                break
        deb=deb+1

bestrun=0
bestscen=""
for l1 in  range(8,30):
    for l2 in range(1,l1):
        for l3 in range(1,l2):
            for solv in ["adam","lbfgs","sgd"]:
                for activ in ["identity","relu"]:
                    scen=str(l1)+"-"+str(l2)+"-"+str(l3)+"-"+solv+"-"+activ
                    print(scen)
                    scen=str(l1)+"-"+str(l2)+"-"+str(l3)+solv+activ
                    scaler = StandardScaler()  
                    cutl=10
                    cutr=cutl-1
                    XX=[ask,bid,mid,diffs,differ(ask,1),differ(bid,1),differ(mid,1),differ(diffs,1),differ(ask,10),differ(bid,10),differ(mid,10),differ(diffs,10),nasks,nbids,differ(nasks,1),differ(nbids,1)]
                    XY=np.column_stack((ask,bid,mid,diffs,volsum,voldif,ndir,differ(ask,1),differ(bid,1),differ(mid,1),differ(diffs,1),differ(ndir,1)))
                    #print(XY.shape)
                    X=XY[cutr:,:]
                    #print(X.shape)
                    mav=moving_average(targ,cutl)
                    Y=classer(mav)
                    clf = MLPClassifier(solver=solv, activation=activ,max_iter=50000, verbose=False,alpha=1e-5,hidden_layer_sizes=(l1,l2,l3), random_state=1,tol=0.0001)
                    scaler.fit(X)  
                    X_train = scaler.transform(X)
                    clf.fit(X_train, Y)
                    #print(clf.score)
                    scaler.fit(XY)  
                    X_pred = scaler.transform(XY)
                    pred=clf.predict(X_pred)*0.29
                    mse = mean_squared_error(pred,targ)
                    base = mean_squared_error(targ,np.array(targ)*0)
                    perf=(1-mse/base)
                    if perf>bestrun:
                        bestrun=perf
                        bestscen=scen
                    if perf>0:
                        print("!!!!!-------IMPROVEMENT------!!!!!!")
                        print(perf)
                    if perf==0:
                        print("zero")
                    if perf<0:
                        print("negative")
                    print("BEST SO FAR")
                    print(bestscen)
                    print(bestrun)

print("best run")
print(bestrun)
print(bestscen)

#mav=30
#lim=mav-1
#print(targ)
#print(moving_average(targ,mav))
#print(targ[lim:]-moving_average(targ,mav))
#err=targ[mav:]-moving_average(targ,mav)[:-1]
#errsq=np.square(err).sum()
#origsq=np.square(targ[mav:]).sum()
#print(1-errsq/origsq)

# Plot the data
#plt.plot(targ[lim:], label='linear')
#plt.plot(moving_average(targ,mav)[:-1], label='linear')

# Add a legend
#plt.legend()

# Show the plot
#plt.show()
