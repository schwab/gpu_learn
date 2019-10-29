import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        # initialize weight matrix and store learning rate
        self.W = np.random.rand(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # the step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # insert a column of 1's a the last entry 
        # this trick allows us to treat bias as though it's a trainable parameter.
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X,y):
                # calcuate the dot product between the input features
                # and the weight matrix
                p = self.step(np.dot(x, self.W))
                # only perform weight update IF the prediction
                # does NOT match the target
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)

        if addBias:
            # insert a column of 1's as the last entry
            X = np.c_[X, np.ones((X.shape[0]))]
        # calculate the dot product of the input features and the weight matrix and call step
        return self.step(np.dot(X, self.W))
