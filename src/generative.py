import numpy as np
import matplotlib.pyplot as plt

class Generative:
    def __init__(self, q, p):


        self.q = q 
        self.p = p 

        if self.p == 2:
            self.data = np.genfromtxt('data/binclassv2.txt',delimiter=',')
        else:
            self.data = np.genfromtxt('data/binclass.txt',delimiter=',')

# Calculates the MLE parameters
    def calcMLE(self, x, posSample, negSample):
        mup = np.mean(x[posSample], axis=0)
        mun = np.mean(x[negSample], axis=0)
        if(self.q != 2):
            k = np.std(x[posSample], axis=0)
        else:
            k = np.std(x)
        sigmap = np.mean(k*k)
        if(self.q != 2):
            k = np.std(x[negSample], axis=0)
        else:
            k = np.std(x)
        sigman = np.mean(k*k)
        mup.reshape((1, mup.shape[0]))
        mun.reshape((1, mun.shape[0]))

        return np.array([mup, mun]), np.array([[sigmap], [sigman]])

    # Will pplot the normal data points
    def plotDataPoints(self, x, y, posSample, negSample):
        plt.plot(x[posSample,0], x[posSample,1], 'r*')
        plt.plot(x[negSample,0], x[negSample,1], 'b*')

    # plot decision boundary
    def plotDecisionBoundary(self, mu, sigma, ki, kj, y, x):
        sigmai = sigma[ki].reshape(())
        sigmaj = sigma[kj].reshape(())

        if self.q == 2:
            plt.title('Same Sigma Part: ' + str(self.p))
            # sigmaj = sigmai
        else:
            plt.title('Different Sigma Part: ' + str(self.p))

        mui = mu[ki].reshape((mu[ki].shape[0], 1))
        muj = mu[kj].reshape((mu[kj].shape[0], 1))

        X, Y = np.meshgrid(x, y)
        Z1 = (sigmai**(-1*mui.shape[0]/2))*np.exp(((X-mui[0])**2 + (Y-mui[1])**2)/(-2*sigmai))
        Z2 = (sigmaj**(-1*mui.shape[0]/2))*np.exp(((X-muj[0])**2 + (Y-muj[1])**2)/(-2*sigmaj))
        Z = Z1 - Z2
        plt.contour(X, Y, Z, 0)

    def main(self):
        x = self.data[:,:self.data.shape[1]-1]
        y = self.data[:,self.data.shape[1]-1]
        posSample = y>0
        negSample = y<0
        x2 = np.arange(np.min(x[:,1]),np.max(x[:,1]),0.05)
        x1 = np.arange(np.min(x[:,0]),np.max(x[:,0]),0.05)

        mu, sigma = self.calcMLE(x, posSample, negSample)

        plt.xlabel('x1')
        plt.ylabel('x2')
        self.plotDataPoints(x, y, posSample, negSample)
        self.plotDecisionBoundary(mu, sigma, 0, 1, x2, x1)

        plt.savefig("output/Q_"+str(self.q)+"_part_"+str(self.p))
        plt.close()



model = Generative(1,1)
model.main()
model = Generative(1,2)
model.main()

model = Generative(2,1)
model.main()
model = Generative(2,2)
model.main()

