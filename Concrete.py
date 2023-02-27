from MWM import *

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
        k = NT(k,g,10000,LL,y,alpha,depth)
    return Wa,Wb,k

if __name__ == "__main__":

    #data preparation
    raw_data = pd.read_csv('peakstress.csv')
    n = len(raw_data)
    x = raw_data["depth"]
    y = raw_data["peakstress"]
    z = raw_data["alpha"]

    y_train40 = []
    y_train41 = []
    y_train42 = []
    y_train43 = []

    for i in range(110):
        if x[i] == 0.4:
            if z[i] == 0:
                y_train40.append(y[i])            
            if z[i] == 0.075:
                y_train41.append(y[i])
            if z[i] == 0.15:
                y_train42.append(y[i])
            if z[i] == 0.3:
                y_train43.append(y[i])           
    y_train40.sort()
    y_train41.sort()
    y_train42.sort()
    y_train43.sort()

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

    with open("41.csv",mode="w",encoding="utf-8-sig",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_list)
        writer.writerows(data_list)

    






    
