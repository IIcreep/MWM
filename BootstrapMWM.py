from __future__ import division
from cmath import nan, pi
from re import S
from unittest import result
import pandas as pd
import numpy as np
import numdifftools as nd
import random
import matplotlib.pyplot as plt
import sympy
from scipy import integrate, interpolate
import copy
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import resample
from arch.bootstrap import CircularBlockBootstrap
import math
import csv

import os
import matplotlib.image as mpimg

import warnings
warnings.filterwarnings('ignore')

def inter(y,m):
    if len(y) == 1:
        return y
    else:
        ran1 = np.random.random(1)
        pos = int(ran1 * len(y))
        ran = np.random.random(m)
        if pos<len(y)-1 and pos >0:
            if abs(y[pos] - y[pos-1]) <= abs(y[pos] - y[pos+1]):
                y.append(y[pos] + ran[0] * (y[pos-1] - y[pos]))
                if m != 1:
                    if abs(y[pos] - y[pos+1]) <= abs(y[pos] - y[pos-2]):
                        y.append(y[pos] + ran[1] * (y[pos+1] - y[pos]))
                    else:
                        y.append(y[pos] + ran[1] * (y[pos-2] - y[pos]))
            else:
                y.append(y[pos] + ran[0] * (y[pos+1] - y[pos]))
                if m != 1:
                    if abs(y[pos] - y[pos-1]) <= abs(y[pos] - y[pos+2]):
                        y.append(y[pos] + ran[1] * (y[pos-1] - y[pos]))
                    else:
                        y.append(y[pos] + ran[1] * (y[pos+2] - y[pos]))
        elif pos == 0:
            y.append(y[pos] + ran[0] * (y[pos+1] - y[pos]))
            if m != 1:
                y.append(y[pos] + ran[1] * (y[pos+2] - y[pos]))
        elif pos ==len(y)-1:
            y.append(y[pos] + ran[0] * (y[pos-1] - y[pos]))
            if m != 1:
                y.append(y[pos] + ran[1] * (y[pos-2] - y[pos]))
        y.sort()
        return y

def smote(y,m):
    names = locals()
    testcop = y
    y = np.array(y).reshape(-1,1)
    model = AgglomerativeClustering(n_clusters=m)
    yhat = model.fit_predict(y)
    for i in range(m):
        names['tem' + str(i)] = []

    for i in range(len(y)):
        for j in range(m):
            if yhat[i] == j:
                names['tem' + str(j)].append(testcop[i])

    length = len(names['tem' + str(0)])
    for i in range(m):
        print(names['tem' + str(i)])

    if len(y) < 10:
        d = names['tem' + str(1)][0] -names['tem' + str(0)][-1]
        for i in range(m-1):
            names['tem' + str(i)] = inter(names['tem' + str(i)],2)
        names['tem' + str(m-1)] = inter(names['tem' + str(m-1)],2)
        names['tem' + str(m-1)].append((names['tem' + str(0)][-1] + names['tem' + str(1)][0])/2)
    for i in range(1,m):
        names['tem' + str(0)].extend(names['tem' + str(i)])
    names['tem' + str(0)].sort()

    ########散点图
    num = 0
    for i in names['tem' + str(0)]:
        if testcop.count(i) != 0:
            plt.scatter(num,i,color='r')
        else:
            plt.scatter(num,i,color='b')
        num = num+1
    plt.xlabel("Serial number of samples")
    plt.ylabel("Nominal tensile strength")
    i = 0
    while True:
        i += 1
        newname = '{}{:d}.png'.format('/data/Serial', i)
        if os.path.exists(newname):
            continue
        plt.savefig(newname,dpi=1000)
        break
    plt.show()

    return names['tem' + str(0)] ,length

def CircularBlockBootstrap(y,length):
    alldata = []
    rs = np.random.RandomState(1234)
    y = np.array(y)
    bs = CircularBlockBootstrap(length,y)
    for data in bs.bootstrap(20):
        data[0][0].sort()
        alldata.append(data[0][0])

    return alldata

def partial_function(f___,input,pos,value,y,alpha,depth):
    tmp  = input[pos,0]
    input[pos,0] = value
    ret = f___(input[0,0],input[1,0],input[2,0],input[3,0],input[4,0],input[5,0],y,alpha,depth)
    input[pos,0] = tmp

    return ret

def partial_derivative(f,input,y,alpha,depth):
    ret = np.empty(len(input))
    for i in range(len(input)):
        fg = lambda x:partial_function(f,input,i,x,y,alpha,depth)
        ret[i] = nd.Derivative(fg)(input[i,0])

    return ret

def LL(c1,a1,z1,c2,a2,z2,y,alpha,depth):
    V = 2.4 * depth * 0.4 * depth
    L = 0
    for i in range(len(y)):
        com1 = V/(2*(16.364+1)**2) * 16.364/ (c1-a1/(1+np.exp(-alpha))) * ((y[i]-z1)/(c1-a1/(1+np.exp(-alpha)))) ** (16.364-1) * np.exp(-((y[i]-z1)/(c1-a1/(1+np.exp(-alpha))))**16.364 * V/(2*(16.364+1)**2))
        com2 = V/(2*(16.364+1)**2) * 16.364/ (c2-a2/(1+np.exp(-alpha))) * ((y[i]-z2)/(c2-a2/(1+np.exp(-alpha)))) ** (16.364-1) * np.exp(-((y[i]-z2)/(c2-a2/(1+np.exp(-alpha))))**16.364 * V/(2*(16.364+1)**2))
        L = L + names['wa' + str(i)] * np.log(com1/names['wa' + str(i)]) + names['wb' + str(i)] * np.log(com2/names['wb' + str(i)])
    return L

def NT(input,g,step,LL,y,alpha,depth):
    X = input
    G = g
    B = np.identity(6)

    for i in range(step):
        print("梯度 ：",np.linalg.norm(G,1))
        if np.linalg.norm(G,1) <= 1:
            break

        print("第 ",i+1,"次")
        a = 1
        bili = 0.01

        d = -np.matmul(np.linalg.inv(B),G)

        print("G : ",G)
        for i in G:
            for j in i:
                if math.isnan(j):
                    return X

        Xcurr = X
        Xnext = Xcurr + bili * d
        fCurr = LL(X[0,0],X[1,0],X[2,0],X[3,0],X[4,0],X[5,0],y,alpha,depth)
        fNext = LL(Xnext[0,0],Xnext[1,0],Xnext[2,0],Xnext[3,0],Xnext[4,0],Xnext[5,0],y,alpha,depth)

        print("np.matmul(G.T,d) : ",np.matmul(d.T,G))
        while True:
            if (Xnext[0,0]-Xnext[1,0]/(1+np.exp(-alpha))) > 0 and (Xnext[3,0]-Xnext[4,0]/(1+np.exp(-alpha))) > 0 and Xnext[2,0]>=0 and Xnext[5,0]>=0:
                break
            a = a + 1
            bili = 0.01 ** a
            Xnext = Xcurr + bili * d
            fNext = LL(Xnext[0,0],Xnext[1,0],Xnext[2,0],Xnext[3,0],Xnext[4,0],Xnext[5,0],y,alpha,depth) 

        print("fCurr : ",fCurr)
        print("fNext : ",fNext)
        g0,g1,g2,g3,g4,g5 = partial_derivative(LL,Xnext,y,alpha,depth)
        G2 = [[g0],[g1],[g2],[g3],[g4],[g5]]
        G2 = np.matrix(G2)
        print("Xcurr : ",Xcurr)
        S = Xnext - Xcurr
        Y = G2 - G

        term1 = np.matmul(Y,Y.T)/np.matmul(Y.T,S)
        term2 = np.matmul(np.matmul(B,S),np.matmul(S.T,B))
        term3 = np.matmul(np.matmul(S.T,B),S)
        B = B + term1 - term2/term3

        X = Xnext
        G = G2

    return X

names = locals()
def main(y,alpha,depth):
    V = 2.4 * depth * 0.4 * depth

    k = [[6],[2],[0],[6],[2],[0]]
    k = np.matrix(k)

    Wa = 0.4
    Wb = 0.6

    for i in range(5):
        for i in range(len(y)):
            por1 = V/(2*(16.364+1)**2) * 16.364 / (k[0,0]-k[1,0]/(1+np.exp(-alpha))) * ((y[i]-k[2,0])/(k[0,0]-k[1,0]/(1+np.exp(-alpha)))) ** (16.364-1) * np.exp(-((y[i]-k[2,0])/(k[0,0]-k[1,0]/(1+np.exp(-alpha))))**16.364 * V/(2*(16.364+1)**2))
            por2 = V/(2*(16.364+1)**2) * 16.364 / (k[3,0]-k[4,0]/(1+np.exp(-alpha))) * ((y[i]-k[5,0])/(k[3,0]-k[4,0]/(1+np.exp(-alpha)))) ** (16.364-1) * np.exp(-((y[i]-k[5,0])/(k[3,0]-k[4,0]/(1+np.exp(-alpha))))**16.364 * V/(2*(16.364+1)**2))
            por1 = Wa   * por1
            por2 = Wb   * por2          
            names['wa' + str(i)] = por1/(por1+por2)
            names['wb' + str(i)] = por2/(por1+por2)            
        Wa = 0
        Wb = 0
        for i in range(len(y)):
            Wa = Wa + names['wa' + str(i)]
        Wa = Wa/len(y)
        print("Wa的值 ：",Wa)

        for i in range(len(y)):
            Wb = Wb + names['wb' + str(i)]
        Wb = Wb/len(y)
        print("Wb的值 ：",Wb)

        dk0,dk1,dk2,dk3,dk4,dk5 = partial_derivative(LL,k,y,alpha,depth)
        g = [[dk0],[dk1],[dk2],[dk3],[dk4],[dk5]]
        g = np.matrix(g)
        print("G ",g)
        k = NT(k,g,5000,LL,y,alpha,depth)
    return Wa,Wb,k

def result(num,k,wa,wb,alpha,depth):
    V = 2.4 * depth * 0.4 * depth
    por1 = V/(2*(16.364+1)**2) * 16.364 / (k[0,0]-k[1,0]/(1+np.exp(-alpha))) * ((num-k[2,0])/(k[0,0]-k[1,0]/(1+np.exp(-alpha)))) ** (16.364-1) * np.exp(-((num-k[2,0])/(k[0,0]-k[1,0]/(1+np.exp(-alpha))))**16.364 * V/(2*(16.364+1)**2))
    por2 = V/(2*(16.364+1)**2) * 16.364 / (k[3,0]-k[4,0]/(1+np.exp(-alpha))) * ((num-k[5,0])/(k[3,0]-k[4,0]/(1+np.exp(-alpha)))) ** (16.364-1) * np.exp(-((num-k[5,0])/(k[3,0]-k[4,0]/(1+np.exp(-alpha))))**16.364 * V/(2*(16.364+1)**2))
    por1 = wa    * por1
    por2 = wb    * por2

    return (por1 + por2)

if __name__ == "__main__":
    raw_data = pd.read_csv('/Users/hu/Desktop/NNI/peakstress.csv')
    n = len(raw_data)
    x = raw_data["depth"]
    y = raw_data["peakstress"]
    z = raw_data["alpha"]

    y_train41 = []

    for i in range(110):
        if x[i] == 0.4:      
            if z[i] == 0.075:
                y_train41.append(y[i])       
    y_train41.sort()

    d = []
    for i in range(10):
        y,length = smote(y_train41,2)
        d.append(CircularBlockBootstrap(y,length))

    header_list = ["wa","wb","input0","input1","input2","input3","input4","input5"]
    data_list = []

    i = 0
    for box in d:
        for item in box:
            data_list.append([])
            fwa,fwb,finput = main(item,0.075,0.4)
            data_list[i].append(fwa)
            data_list[i].append(fwb)
            data_list[i].append(float(finput[0]))
            data_list[i].append(float(finput[1]))
            data_list[i].append(float(finput[2]))
            data_list[i].append(float(finput[3]))
            data_list[i].append(float(finput[4]))
            data_list[i].append(float(finput[5]))
            i = i + 1

    with open("/Users/hu/Desktop/NNI/data/41.csv",mode="w",encoding="utf-8-sig",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_list)
        writer.writerows(data_list)

    






    
