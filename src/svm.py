import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


class SVM:
    def __init__(self, p):
        self.p = p

        if self.p == 2:
            self.data = np.genfromtxt('data/binclassv2.txt',delimiter=',')
        else:
            self.data = np.genfromtxt('data/binclass.txt',delimiter=',')

    def plotDataPoints(self, x, y, posSample, negSample):
        plt.plot(x[posSample,0], x[posSample,1], 'r*')
        plt.plot(x[negSample,0], x[negSample,1], 'b*')

    def plotDecisionBoundary(self, clf, y, x):
        X, Y = np.meshgrid(x, y)   
        Z = clf.predict(np.c_[X.ravel(), Y.ravel()])
        Z = Z.reshape(X.shape)
        plt.contour(X, Y, Z)

    def main(self):
        x = self.data[:,:self.data.shape[1]-1]
        y = self.data[:,self.data.shape[1]-1]
        posSample = y>0
        negSample = y<0
        x2 = np.arange(np.min(x[:,1]),np.max(x[:,1]),0.05)
        x1 = np.arange(np.min(x[:,0]),np.max(x[:,0]),0.05)

        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(x, y)

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('SVM Part: ' + str(self.p))
        self.plotDataPoints(x, y, posSample, negSample)
        self.plotDecisionBoundary(clf, x2, x1)
        plt.savefig("output/svm_part_"+str(self.p))
        plt.close()

    
model = SVM(2)
model.main()
model = SVM(1)
model.main()
