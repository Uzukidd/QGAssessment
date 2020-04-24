import numpy as np

class network(object) :

    class costFun():
        MSE = 0
        CEE = 1

    def __init__(self, layers) :

        """Inits Data with the size of layers in list format."""

        self.sizes = layers

        self.size_length = len(self.sizes)

        self.init()

    def init(self) : 

	    self.biases = [np.random.rand(self.sizes[i]) for i in range(0, self.size_length)]

	    self.weight = [np.random.rand(self.sizes[i], 1 if i == 0 else self.sizes[i - 1]) for i in range(0, self.size_length)]


    def feedForward(self, input) :

        res = [
            np.zeros(self.sizes[i]) for i in range(0, self.size_length)
        ]

        res[0] = input

        for i in range(1, self.size_length) :

            res[i] = np.array([self.sigmoid(res[i - 1], w, b) for w, b in zip(self.weight[i], self.biases[i])])


        return res[self.size_length - 1]


    def backProp(self, input, ans, eta, cost, lamda) : 

        # Firstly just copy the code above

        res = np.array([
            np.zeros(self.sizes[i]) for i in range(0, self.size_length)
        ])

        zs = np.array([
            np.zeros(self.sizes[i]) for i in range(0, self.size_length)
        ])

        res[0] = input

        for i in range(1, self.size_length) :

            res[i] = np.array([self.sigmoid(res[i - 1], w, b) for w, b in zip(self.weight[i], self.biases[i])])

            zs[i] += np.array([res[i - 1].dot(w) + b for w, b in zip(self.weight[i], self.biases[i])])



        # Nabla_w's array distribution is the same to the weight
        # So nabla_w[x][y][z] is the nabla of weight[x][y][z]
        # So as delta and biases

        nabla_w = [np.zeros((self.sizes[i], 1 if i == 0 else self.sizes[i - 1])) for i in range(0, self.size_length)]

        delta =  [np.zeros(self.sizes[i]) for i in range(0, self.size_length)]




        if(cost == self.costFun.MSE) :

            delta[self.size_length - 1] += res[self.size_length - 1] - ans

            delta[self.size_length - 1] *= self.sigmoid_Prime(zs[self.size_length - 1])

        elif(cost == self.costFun.CEE) :

            delta[self.size_length - 1] += res[self.size_length - 1] - ans


        for i in range(0, self.sizes[self.size_length - 1]) :

        	nabla_w[self.size_length - 1][i] = res[self.size_length - 2] * delta[self.size_length - 1][i]

        # To perform iteration
        # from second last layer to second layer
        
        for k in range(1, self.size_length - 1) :

            l = self.size_length - k - 1

            delta[l] = np.zeros(self.sizes[l])

            # l: The layer number now we are.

            for i in range(0, self.sizes[l]) :

                for m in range(0, self.sizes[l + 1]) :

                    delta[l][i] += delta[l + 1][m] * self.weight[l + 1][m][i]
                
            delta[l] *= self.sigmoid_Prime(zs[l])
            
            for i in range(0, self.sizes[l]) :
                
                nabla_w[l][i] = res[l - 1] * delta[l][i]


        # Update every w and b

        for k in range(0, self.size_length - 1) :

            l = self.size_length - k - 1

            for i in range(0, self.sizes[l]) :

                self.weight[l][i] -= (eta * nabla_w[l][i] + lamda * self.weight[l][i])

                self.biases[l][i] -= (eta * delta[l][i])

    def sigmoid(self, x, w, b) :

        temp = x.dot(w) + b

        if(temp >= 0) :
            return 1.0 / (1.0 + np.exp(-temp))
        else:
            return np.exp(temp) / (1.0 + np.exp(temp))

    def sigmoid_Prime(self, z) :


        temp = np.array([1.0 / (1.0 + np.exp(-i)) if i >= 0 else np.exp(i) / (1.0 + np.exp(i)) for i in z])

        return temp / (1.0 + temp)
