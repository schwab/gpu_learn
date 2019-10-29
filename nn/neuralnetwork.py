import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start by looping from the first layer but stop before the last 2 layers
        for i in np.arange(0, len(layers) -2):
            # init weights randomly
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            self.W.append(w /np.sqrt(layers[i]))
        # handle the last 2 special layers wher the input
        # needs a bias term but the o/p does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # sigmoid activation for x
        return 1.0/(1+np.exp(-x))

    def sigmod_deriv(self,x):
        # compute the derivative of the sigmoid function
        # assumes x has already been passed through self.sigmoid(x)
        return x * (1-x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        X = np.c_[X, np.ones((X.shape[0]))]
        # loop for each epoch
        for epoch in np.arange(0, epochs):
            # train network for each datapoint
            for (x, target) in zip(X,y):
                self.fit_partial(x, target)
            if epoch == 0 or (epoch +1) % displayUpdate ==0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        # creaet list of output activation for each layer
        A = [np.atleast_2d(x)]
        # FEEDFORWARD
        for layer in np.arange(0, len(self.W)):
            # feedforward activation for the current layer
            # by taking the dot product between the activation
            # and the weight matrix "net input" of the current layer
            net = A[layer].dot(self.W[layer])

            # netoutput
            out = self.sigmoid(net)

            # add to the list of activations
            A.append(out)
        
        # BACKPROPAGATION
        # compute difference between prediction and true target value
        error = A[-1] - y

        #apply the chain rule to build a list of deltas
        D = [error * self.sigmod_deriv(A[-1])]

        # loop over the layers in reverse order (ignoring last 2)
        for layer in np.arange(len(A)-2, 0, -1):
            # delta of current layer = delta of last layer dotted with the weight matrix of the current layer
            # followed by muliplying the delta by the derivative of the non-linear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmod_deriv(A[layer])
            D.append(delta)
        # reverse delta 
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        # loop over layers
        for layer in np.arange(0, len(self.W)):
            # update weights by taking the dot product of the layer activations
            # with their respective deltas and * by a small learning rate 
            # add to weight matrix ie "learning"
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
            
    def predict(self, X, addBias=True):
        # create op prediction
        p = np.atleast_2d(X)
        # check if bias column should be added
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        # loop over layers
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss


