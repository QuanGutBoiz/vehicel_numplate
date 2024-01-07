from utils import *
import seaborn as sns
import numpy as np 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import pandas as pd
class Model:
    def __init__(self, layers = [2352, 128, 64, 36]):
        self.layers = layers
        self.params = self.init_params()
        self.num_layers = len(layers)
        self.activations = {'relu': relu, 'softmax': softmax}
        self.derivatives = {'relu': drelu}
        self.loss = cross_entropy
        self.train_acc_history = []
        self.test_acc_history = []
        self.loss_history = []
        self.target_names=self.init_target()
    def init_params(self):
        params = {}
        for i in range(1, len(self.layers)):
            params['W' + str(i)] = np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(2 / self.layers[i-1])
            params['b' + str(i)] = np.zeros((1, self.layers[i]))
        return params
    def init_target(self):
        target_names=[]
        for i in range(36):
            target_names.append('class '+str(i))
            
        return target_names
    def forward(self, X):
        cache = {'A0': X}
        for i in range(1, len(self.layers)):
            cache['Z' + str(i)] = np.dot(cache['A' + str(i-1)], self.params['W' + str(i)]) + self.params['b' + str(i)]
            if i == len(self.layers) - 1:
                cache['A' + str(i)] = self.activations['softmax'](cache['Z' + str(i)])
            else:
                cache['A' + str(i)] = self.activations['relu'](cache['Z' + str(i)])
        return cache['A' + str(len(self.layers) - 1)], cache
    
    def backward(self, y, y_hat, cache):
        grads = {}
        dA = y_hat - y
        for i in reversed(range(1, len(self.layers))):
            dZ = dA * self.derivatives['relu'](cache['Z' + str(i)])
            grads['W' + str(i)] = np.dot(cache['A' + str(i-1)].T, dZ) / y.shape[0]
            grads['b' + str(i)] = np.sum(dZ, axis=0, keepdims=True) / y.shape[0]
            dA = np.dot(dZ, self.params['W' + str(i)].T)
        return grads
    
    def update(self, grads, lr):
        for i in range(1, len(self.layers)):
            self.params['W' + str(i)] -= lr * grads['W' + str(i)]
            self.params['b' + str(i)] -= lr * grads['b' + str(i)]


    def fit(self, X_train, y_train, X_test, y_test, epochs, lr, print = True):
        for epoch in range(epochs):
            y_hat, cache = self.forward(X_train)
            loss = self.loss(y_train, y_hat)
            self.loss_history.append(np.mean(loss))
            grads = self.backward(y_train, y_hat, cache)
            self.update(grads, lr)
            if print:
                print(f'Epoch: {epoch+1} / {epochs}, Loss: {np.mean(loss): .4f}')
                print('Train Accuracy: {}'.format(self.evaluate(X_train, np.argmax(y_train,axis=1))))
                self.train_acc_history.append(self.evaluate(X_train, np.argmax(y_train, axis=1)))
                print('Test Accuracy: {}'.format(self.evaluate(X_test, np.argmax(y_test, axis=1))))
                self.test_acc_history.append(self.evaluate(X_test, np.argmax(y_test, axis=1)))
    def show(self,X_train,y_train,X_test,y_test,show=True):
        his_y=[]
        for i in range(len(y_test)):
            his_y.append(self.predict(X_test[i]))
        
        if show:

            report=classification_report(np.argmax(y_test,axis=1), his_y, target_names=self.target_names,output_dict=True)
            print(report)
            final=pd.DataFrame(report).transpose()
            print(final)
        else:

            cm = confusion_matrix(np.argmax(y_test,axis=1),his_y)
            # print(cm)
            sns.heatmap(cm,cmap='Blues',cbar=False,annot=True)
            plt.xticks(np.arange(36)+0.5,np.arange(36))
            plt.yticks(np.arange(36)+0.5,np.arange(36))
            plt.xlabel('Predict Classes')
            plt.ylabel('Actual Classes')
            plt.show()
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[True,False])
        # disp.plot()
        # plt.show()  
    def predict(self, X):
        y_hat, _ = self.forward(X)
        return np.argmax(y_hat, axis=1)
    

    def evaluate(self, X, y):
        y_hat, _ = self.forward(X)
        y_hat = np.argmax(y_hat, axis=1)
        return np.sum(y_hat == y) / y.shape[0]
    
    
    def save(self, path):
        np.save(path, self.params)
    
    def load(self, path):
        self.params = np.load(path, allow_pickle=True).item()
    

