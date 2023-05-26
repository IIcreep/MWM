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
        for i in range(m-1):
            names['tem' + str(i)] = inter(names['tem' + str(i)],2)
        names['tem' + str(m-1)] = inter(names['tem' + str(m-1)],2)
    for i in range(1,m):
        names['tem' + str(0)].extend(names['tem' + str(i)])
    names['tem' + str(0)].sort()

    return names['tem' + str(0)] ,length

def CCircularBlockBootstrap(y,length,reps):
    alldata = []
    y = np.array(y)
    bs = CircularBlockBootstrap(length,y)
    for data in bs.bootstrap(reps):
        resampled_y = data[0][0]
        resampled_y.sort()
        alldata.append(resampled_y)

    return alldata

def partial_function(f___,input,pos,value,y,depth,alpha):
    tmp  = input[pos,0]
    input[pos,0] = value
    ret = f___(input[0,0],input[1,0],input[2,0],y,depth,alpha)
    input[pos,0] = tmp

    return ret

def partial_derivative(f,input,y,depth,alpha):
    ret = np.empty(len(input))
    for i in range(len(input)):
        fg = lambda x:partial_function(f,input,i,x,y,depth,alpha)
        ret[i] = nd.Derivative(fg)(input[i,0])

    return ret

def LL(k0,k1,k2,y,depth,alpha):
    L = 0
    for i in range(len(y)):
        P = 2 * 40 * depth /3 /2.176 * y[i]
        Y = 1.12 + 0.23 * alpha - 0.097 * alpha ** 2 + 0.038 * alpha ** 3
        K = (P * alpha**2) / Y
        V = (2 * 4.626**2)/(2.4 * depth * 40 * depth)

        com11 = np.exp(- ((y[i]-k1)**3.626)/(k0**3.626 * V * K)) / k0**3.626 / V * (y[i]-k1)**2.626
        com12 = 3.626/K - K**-2 * (y[i]-k1) * alpha**2/Y * (2 * 40 * depth /3 /2.176)
        com1 = com11 * com12
        com21 = np.exp(- ((y[i]-k1)**3.626)/(k2**3.626 * V * K)) / k2**3.626 / V * (y[i]-k1)**2.626
        com22 = 3.626/K - K**-2 * (y[i]-k1) * alpha**2/Y * (2 * 40 * depth /3 /2.176)
        com2 = com21 * com22

        L = L + names['wa' + str(i)] * np.log(com1/names['wa' + str(i)]) + names['wb' + str(i)] * np.log(com2/names['wb' + str(i)])
    return L

def NT(input,g,step,LL,y,depth,alpha):
    X = input
    G = g
    B = np.identity(3)

    for i in range(step):
        print("梯度 ：",np.linalg.norm(G,1))
        if np.linalg.norm(G,1) <= 1.e-1:
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
        fCurr = LL(X[0,0],X[1,0],X[2,0],y,depth,alpha)
        fNext = LL(Xnext[0,0],Xnext[1,0],Xnext[2,0],y,depth,alpha)

        print("np.matmul(G.T,d) : ",np.matmul(d.T,G))
        while True:
            if Xnext[0,0] > 0 and Xnext[1,0] > 0 and Xnext[2,0]>=0:
                break
            a = a + 1
            bili = 0.01 ** a
            Xnext = Xcurr + bili * d
            fNext = LL(Xnext[0,0],Xnext[1,0],Xnext[2,0],y,depth,alpha) 

        print("fCurr : ",fCurr)
        print("fNext : ",fNext)
        g0,g1,g2 = partial_derivative(LL,Xnext,y,depth,alpha)
        G2 = [[g0],[g1],[g2]]
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
def main(y,depth,alpha):
    k = [[4],[1],[4]]
    k = np.matrix(k)

    Wa = 0.6
    Wb = 0.4

    for i in range(5):
        
        for i in range(len(y)):
            P = 2 * 40 * depth /3 /2.176 * y[i]
            Y = 1.12 + 0.23 * alpha - 0.097 * alpha ** 2 + 0.038 * alpha ** 3
            K = (P * alpha**2) / Y
            V = (2 * 4.626**2)/(2.4 * depth * 40 * depth)

            por11 = np.exp(- ((y[i]-k[1,0])**3.626)/(k[0,0]**3.626 * V * K)) / k[0,0]**3.626 / V * (y[i]-k[1,0])**2.626
            por12 = 3.626/K - K**-2 * (y[i]-k[1,0]) * alpha**2/Y * (2 * 40 * depth /3 /2.176)
            por1 = por11 * por12
            por21 = np.exp(- ((y[i]-k[1,0])**3.626)/(k[2,0]**3.626 * V * K)) / k[2,0]**3.626 / V * (y[i]-k[1,0])**2.626
            por22 = 3.626/K - K**-2 * (y[i]-k[1,0]) * alpha**2/Y * (2 * 40 * depth /3 /2.176)
            por2 = por21 * por22
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

        dk0,dk1,dk2= partial_derivative(LL,k,y,depth,alpha)
        g = [[dk0],[dk1],[dk2]]
        g = np.matrix(g)
        print("G ",g)
        k = NT(k,g,5000,LL,y,depth,alpha)
    return Wa,Wb,k

if __name__ == "__main__":

    #data preparation
    raw_data = pd.read_csv('/Users/hu/Desktop/NNI/混泥土3点弯的数据/peakstress.csv')
    n = len(raw_data)
    x = raw_data["depth"]
    y = raw_data["peakstress"]
    z = raw_data["alpha"]

    y_train40 = []
    y_train41 = []
    y_train42 = []
    y_train43 = []
    y_train90 = []
    y_train91 = []
    y_train92 = []
    y_train93 = []
    y_train20 = []
    y_train21 = []
    y_train22 = []
    y_train23 = []
    y_train50 = []
    y_train51 = []
    y_train52 = []
    y_train53 = []

    for i in range(n):
        if x[i] == 0.4:
            if z[i] == 0:
                y_train40.append(y[i])            
            if z[i] == 0.075:
                y_train41.append(y[i])
            if z[i] == 0.15:
                y_train42.append(y[i])
            if z[i] == 0.3:
                y_train43.append(y[i])
        if x[i] == 0.93:
            if z[i] == 0:
                y_train90.append(y[i])            
            if z[i] == 0.075:
                y_train91.append(y[i])
            if z[i] == 0.15:
                y_train92.append(y[i])
            if z[i] == 0.3:
                y_train93.append(y[i])   
        if x[i] == 2.15:
            if z[i] == 0:
                y_train20.append(y[i])            
            if z[i] == 0.075:
                y_train21.append(y[i])
            if z[i] == 0.15:
                y_train22.append(y[i])
            if z[i] == 0.3:
                y_train23.append(y[i])      
        if x[i] == 5:
            if z[i] == 0:
                y_train50.append(y[i])            
            if z[i] == 0.075:
                y_train51.append(y[i])
            if z[i] == 0.15:
                y_train52.append(y[i])
            if z[i] == 0.3:
                y_train53.append(y[i])      
    y_train40.sort()
    y_train41.sort()
    y_train42.sort()
    y_train43.sort()
    y_train90.sort()
    y_train91.sort()
    y_train92.sort()
    y_train93.sort()
    y_train20.sort()
    y_train21.sort()
    y_train22.sort()
    y_train23.sort()
    y_train50.sort()
    y_train51.sort()
    y_train52.sort()
    y_train53.sort()

    reps = 10
    resampled_data = []
    y,length = smote(y_train53,2)
    resampled_data = CCircularBlockBootstrap(y,length,reps)

    # print(resampled_data)

    header_list = ["wa","wb","input0","input1","input2"]
    data_list = []

    i = 0
    for box in resampled_data:
        data_list.append([])
        fwa,fwb,finput = main(box,500,0.3)
        data_list[i].append(fwa)
        data_list[i].append(fwb)
        data_list[i].append(float(finput[0]))
        data_list[i].append(float(finput[1]))
        data_list[i].append(float(finput[2]))
        i = i + 1

    with open("/Users/hu/Desktop/NNI/data/53.csv",mode="w",encoding="utf-8-sig",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_list)
        writer.writerows(data_list)
