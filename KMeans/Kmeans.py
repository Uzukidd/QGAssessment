import numpy as np

class Kmeans(object) :

    class Distance() :
        Euclidean = 0
        Manhattan = 1

    def __init__(self, count, dis) :

        self.count = count

        self.dis = dis

        self.collective = np.random.rand(count, dis)


    def __itera(self, data, dis = 0, debug = 0) :

        self.__collect(self.classify(data, dis = dis), data, debug = debug)
    
    def fit(self, data, maxIter = 1000, threhold = -1, distance = 0, debug = 0) :

        _debug_counts = 0

        _debug_com = 0

        for i in range(0, maxIter) :

            temp = self.collective.copy()

            self.__itera(data, dis = distance, debug = debug)

            com = (temp == self.collective).sum()

            if(debug): 
                print("It reaches to {0}".format(com))
                print("SSE:{0}".format(Kmeans.SSE(self.collective, data, self.classify(data))))

            _debug_counts = i

            _debug_com = 1

            if(threhold == -1 and com == self.count * 2) : 
                break
            elif(threhold >= 0 and threhold <= com) : 
                break

            _debug_com = 0

        if(debug): 
            print("Had been iterated to {0}".format(_debug_counts))
            if(not _debug_com): print("Not fitted!")


        

    def classify(self, data, dis = 0) :

        result = np.zeros((len(data)))

        for i in range(0, len(data)) :

            if(dis == self.Distance.Euclidean) : result[i] = np.argmin([Kmeans.__Euclidean(data[i], d) for d in self.collective])

            if(dis == self.Distance.Manhattan) : result[i] = np.argmin([Kmeans.__Manhattan(data[i], d) for d in self.collective])

        return result

    def __collect(self, dataClass, data, debug = 0) :

        for i in range(0, self.count) :

            temp = data[dataClass == i]

            if(len(temp)) : self.collective[i] = temp.mean(axis = 0)

    

    @staticmethod
    def __Euclidean(p1, p2) :

        temp = np.power(p1 - p2, 2.0).sum()

        return np.sqrt(temp)

    @staticmethod
    def __Manhattan(p1, p2) :

        return np.abs(p1 - p2).sum(axis = 0)


    @staticmethod
    def SSE(collective, data, dataClass) :

        res = []

        for i in range(0, len(collective)) :

            temp = data[dataClass == i]

            if(len(temp)) : res.append(np.array([np.power(d - collective[i], 2.0) for d in temp]).sum())
            else : res.append(0)

        return res


class biKmeans(object) :

    def __init__(self, count, dis) :

        self.count = count

        self.dis = dis

        self.collective = np.random.rand(count, dis)

    
    def fit(self, data, maxIter = 1000, maxGeneral = 1, threhold = -1, distance = 0, debug = 0) :

        _debug_counts = 0

        _debug_com = 0

        collective_Update = np.random.rand(1, self.dis)

        while(len(collective_Update) < self.count) :


            SSE = Kmeans.SSE(collective_Update, data, biKmeans.classify(collective_Update, data, dis = distance))

            changer = np.argmax(SSE)

            Kmeans_Temp = Kmeans(2, self.dis)

            data_Class = biKmeans.classify(collective_Update, data, dis = distance)

            if(debug) : 
                print("Collective:{0}".format(collective_Update))
                print("SSE:{0}".format(SSE))
                print("changer:{0}".format(changer))
                print("data_Class:{0}".format(data_Class))
                print("stage:{0}".format(len(collective_Update)))

            Kmeans_Temp.fit(data[data_Class == changer], maxIter = maxIter, distance = distance)

            int_temp = len(collective_Update) + 2

            collective_Update = np.append([collective_Update], [Kmeans_Temp.collective])

            collective_Update = np.delete(collective_Update.reshape(int_temp, self.dis), changer, axis = 0)

            if(debug) : 

                print("Temp:{0}".format(Kmeans_Temp.collective))
                print("Collective:{0}".format(collective_Update))
                print("SSE:{0}".format(SSE))
                print("changer:{0}".format(changer))
                print("data_Class:{0}".format(data_Class))
                print("stage:{0}".format(len(collective_Update)))

        Kmeans_Temp = Kmeans(len(collective_Update), self.dis)

        Kmeans_Temp.fit(data, maxIter = maxIter, distance = distance)

        collective_Update = Kmeans_Temp.collective.copy()

        self.collective = collective_Update.copy()

    @staticmethod
    def __Euclidean(p1, p2) :

        temp = np.power(p1 - p2, 2.0).sum()

        return np.sqrt(temp)

    @staticmethod
    def __Manhattan(p1, p2) :

        return np.abs(p1 - p2).sum(axis = 0)
        
    @staticmethod
    def classify(collective, data, dis = 0) :

        result = np.zeros((len(data)))

        for i in range(0, len(data)) :

            if(dis == Kmeans.Distance.Euclidean) : 
                result[i] = np.argmin([biKmeans.__Euclidean(data[i], d) for d in collective])

            if(dis == Kmeans.Distance.Manhattan) : 
                result[i] = np.argmin([biKmeans.__Manhattan(data[i], d) for d in collective])

        return result
