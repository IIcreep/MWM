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

def CircularBlockBootstrap(y,length):
    alldata = []
    rs = np.random.RandomState(1234)
    y = np.array(y)
    bs = CircularBlockBootstrap(length,y,rs)
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

def result(num,k,wa,wb,alpha,depth):
    V = 2.4 * depth * 0.4 * depth
    por1 = V/(2*(16.364+1)**2) * 16.364 / (k[0,0]-k[1,0]/(1+np.exp(-alpha))) * ((num-k[2,0])/(k[0,0]-k[1,0]/(1+np.exp(-alpha)))) ** (16.364-1) * np.exp(-((num-k[2,0])/(k[0,0]-k[1,0]/(1+np.exp(-alpha))))**16.364 * V/(2*(16.364+1)**2))
    por2 = V/(2*(16.364+1)**2) * 16.364 / (k[3,0]-k[4,0]/(1+np.exp(-alpha))) * ((num-k[5,0])/(k[3,0]-k[4,0]/(1+np.exp(-alpha)))) ** (16.364-1) * np.exp(-((num-k[5,0])/(k[3,0]-k[4,0]/(1+np.exp(-alpha))))**16.364 * V/(2*(16.364+1)**2))
    por1 = wa    * por1
    por2 = wb    * por2

    return (por1 + por2)
