import pandas as pd

def E(L,F,B,H,d):
    P = F*2*B*H/3/2.176
    res = (P*L**2)/(4*B*H**3*d*0.1)

    return res

def Ur(E,v):
    res = E/(2*(1+v))*10**-3

    return res

if __name__ == "__main__":
    raw_data = pd.read_csv('/Users/hu/Desktop/NNI/混泥土3点弯的数据/peakstress.csv')
    n = len(raw_data)
    x = raw_data["depth"]
    y = raw_data["peakstress"]
    z = raw_data["alpha"]
    d = raw_data["displacement"]

    y40,y41,y42,y43 = [],[],[],[]
    y90,y91,y92,y93 = [],[],[],[]
    d40,d41,d42,d43 = [],[],[],[]
    d90,d91,d92,d93 = [],[],[],[]
    for i in range(n):
        if x[i] == 0.4:
            if z[i] == 0:
                y40.append(y[i])    
                d40.append(y[i])           
            if z[i] == 0.075:
                y41.append(y[i])
                d41.append(y[i])
            if z[i] == 0.15:
                y42.append(y[i])
                d42.append(y[i])
            if z[i] == 0.3:
                y43.append(y[i])
                d43.append(y[i])
        if x[i] == 0.93:
            if z[i] == 0:
                y90.append(y[i])    
                d90.append(y[i])           
            if z[i] == 0.075:
                y91.append(y[i])
                d91.append(y[i])
            if z[i] == 0.15:
                y92.append(y[i])
                d92.append(y[i])
            if z[i] == 0.3:
                y93.append(y[i])
                d93.append(y[i])

    H = 40
    B = 40            
    L = 2.176*H
    v = 0.3

    E40 = []
    Ur40 = []
    for i in range(len(y40)):
        temE = E(L,y40[i],B,H,d40[i])
        temUr = Ur(temE,v)
        E40.append(temE)
        Ur40.append(temUr)
    E41 = []
    Ur41 = []
    for i in range(len(y41)):
        temE = E(L,y41[i],B,H,d41[i])
        temUr = Ur(temE,v)
        E41.append(temE)
        Ur41.append(temUr)
    E42 = []
    Ur42 = []
    for i in range(len(y42)):
        temE = E(L,y42[i],B,H,d42[i])
        temUr = Ur(temE,v)
        E42.append(temE)
        Ur42.append(temUr)
    E43 = []
    Ur43 = []
    for i in range(len(y43)):
        temE = E(L,y43[i],B,H,d43[i])
        temUr = Ur(temE,v)
        E43.append(temE)
        Ur43.append(temUr)

    H = 93
    B = 40            
    L = 2.176*H

    E90 = []
    Ur90 = []
    for i in range(len(y90)):
        temE = E(L,y90[i],B,H,d90[i])
        temUr = Ur(temE,v)
        E90.append(temE)
        Ur90.append(temUr)
    E91 = []
    Ur91 = []
    for i in range(len(y91)):
        temE = E(L,y91[i],B,H,d91[i])
        temUr = Ur(temE,v)
        E91.append(temE)
        Ur91.append(temUr)
    E92 = []
    Ur92 = []
    for i in range(len(y92)):
        temE = E(L,y92[i],B,H,d92[i])
        temUr = Ur(temE,v)
        E92.append(temE)
        Ur92.append(temUr)
    E93 = []
    Ur93 = []
    for i in range(len(y93)):
        temE = E(L,y93[i],B,H,d93[i])
        temUr = Ur(temE,v)
        E93.append(temE)
        Ur93.append(temUr)
    
