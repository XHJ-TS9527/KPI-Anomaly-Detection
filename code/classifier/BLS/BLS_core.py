import torch
import numpy as np
from sklearn import preprocessing
from numpy import random
import torch.nn.functional as F
import time


class broadnet():

    class scaler():
        def __init__(self):
            self._mean = 0
            self._std = 0

        def fit_transform(self, traindata):
            self._mean = traindata.mean(axis=0)
            self._std = traindata.std(axis=0)
            return (traindata - self._mean) / self._std

        def transform(self, testdata):
            return (testdata - self._mean) / self._std

    class node_generator():
        def __init__(self, whiten=False):
            self.Wlist = []
            self.blist = []
            self.nonlinear = 0
            self.whiten = whiten

        def sigmoid(self, data):
            return 1.0 / (1 + np.exp(-data))

        def linear(self, data):
            return data

        def tanh(self, data):
            return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

        def relu(self, data):
            return np.maximum(data, 0)

        def orth(self, W):
            for i in range(0, W.shape[1]):
                w = np.mat(W[:, i].copy()).T
                w_sum = 0
                for j in range(i):
                    wj = np.mat(W[:, j].copy()).T
                    w_sum += (w.T.dot(wj))[0, 0] * wj
                w -= w_sum
                w = w / np.sqrt(w.T.dot(w))
                W[:, i] = np.ravel(w)
            return W

        def generator(self, shape, times):
            for i in range(times):
                W = 2 * random.random(size=shape) - 1
                if self.whiten == True:
                    W = self.orth(W)
                b = 2 * random.random() - 1
                yield (W, b)

        def generator_nodes(self, data, times, batchsize, nonlinear):
            self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
            self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

            self.nonlinear = {'linear': self.linear,
                              'sigmoid': self.sigmoid,
                              'tanh': self.tanh,
                              'relu': self.relu
                              }[nonlinear]
            nodes = self.nonlinear(data.dot(self.Wlist[0]) + self.blist[0])
            for i in range(1, len(self.Wlist)):
                nodes = np.column_stack((nodes, self.nonlinear(data.dot(self.Wlist[i]) + self.blist[i])))
            return nodes

        def transform(self, testdata):
            testnodes = self.nonlinear(testdata.dot(self.Wlist[0]) + self.blist[0])
            for i in range(1, len(self.Wlist)):
                testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i]) + self.blist[i])))
            return testnodes

        def update(self, otherW, otherb):
            self.Wlist += otherW
            self.blist += otherb

    def __init__(self,
                 maptimes=10,
                 enhencetimes=10,
                 map_function='linear',
                 enhence_function='linear',
                 batchsize='auto',
                 reg=0.001):

        self._maptimes = maptimes
        self._enhencetimes = enhencetimes
        self._batchsize = batchsize
        self._reg = reg
        self._map_function = map_function
        self._enhence_function = enhence_function

        self.W = 0
        self.pesuedoinverse = 0
        self.normalscaler = self.scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.mapping_generator = self.node_generator()
        self.enhence_generator = self.node_generator(whiten=True)

    def fit(self, data, label):
        if self._batchsize == 'auto':
            self._batchsize = data.shape[1]
        data = self.normalscaler.fit_transform(data)
        label = self.onehotencoder.fit_transform(np.mat(label).T)

        mappingdata = self.mapping_generator.generator_nodes(data, self._maptimes, self._batchsize, self._map_function)
        enhencedata = self.enhence_generator.generator_nodes(mappingdata, self._enhencetimes, self._batchsize,
                                                             self._enhence_function)
        inputdata = np.column_stack((mappingdata, enhencedata))
        pesuedoinverse = self.pinv(inputdata, self._reg)
        self.W = pesuedoinverse.dot(label)

    def pinv(self, A, reg):
        return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def decode_proab(self, Y_onehot):
        y = np.array(Y_onehot)
        y = torch.from_numpy(y)
        if torch.cuda.is_available():
            y = y.to('cuda')
        else:
            y = y.to('cpu')
        pred = F.softmax(y,dim=1)
        pred = pred.to('cpu')
        return np.array(pred)

    def accuracy(self, predictlabel, label):
        label = np.ravel(label).tolist()
        predictlabel = predictlabel.tolist()
        count = 0
        for i in range(len(label)):
            if label[i] == predictlabel[i]:
                count += 1
        return (round(count / len(label), 5))

    def predict(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_mappingdata = self.mapping_generator.transform(testdata)
        test_enhencedata = self.enhence_generator.transform(test_mappingdata)
        test_inputdata = np.column_stack((test_mappingdata, test_enhencedata))
        return self.decode(test_inputdata.dot(self.W))

    def predict_proab(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_mappingdata = self.mapping_generator.transform(testdata)
        test_enhencedata = self.enhence_generator.transform(test_mappingdata)
        test_inputdata = np.column_stack((test_mappingdata, test_enhencedata))
        return self.decode_proab(test_inputdata.dot(self.W))[:, 1]


class BLS():

    def train_BLS(self, model, train_X, train_Y):
        before_time = time.time()
        model.fit(train_X, train_Y)
        after_time = time.time()
        return model, after_time - before_time

    def test_BLS(self, model, test_X):
        return model.predict_proab(test_X)
