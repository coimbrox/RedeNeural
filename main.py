import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as datasets

plt.rcParams['figure.figsize'] = [10, 6]
plt.style.use('dark_background')

x,y = datasets.make_moons(n_samples=500,noise=0.05)
x.shape,y.shape

print(f'{x.shape= }, {y.shape= }')

pd.DataFrame({'x_1':x[:,0],'x_2':x[:,1],'y':y})

unique = np.unique(y, return_counts=True)
for label,qt_label in zip(unique[0],unique[1]):
    print(f'Label: {label}\t Counts: {qt_label= }') 

plt.scatter(x[:,0],x[:,1],c=y,s=50,alpha=0.5,cmap='cool')

class NnModel:
    def __init__(self,x:np.ndarray,y:np.ndarray,hidden_neurons:int=10,output_neurons:int=2):
        np.random.seed(8)
        self.x = x
        self.y = y
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.input_neurons = self.x.shape[1]
        
        #inicializando pesos e bias
        #Xavier Initialization -> variancia dos pesos iguais em todas as camadas
        self.W1 = np.random.randn(self.input_neurons,self.hidden_neurons) / np.sqrt(self.input_neurons)
        self.B1 = np.zeros((1,self.hidden_neurons))
        self.W2 = np.random.randn(self.hidden_neurons,self.output_neurons) / np.sqrt(self.hidden_neurons)
        self.B2 = np.zeros((1,self.output_neurons))
        self.model_dict = {'W1': self.W1,'B1': self.B1,'W2': self.W2,'B2': self.B2}
        self.z1 = 0
        self.f1 = 0  
    def foward(self,x:np.ndarray) -> np.ndarray:
        #eq da reta
        self.z1 = x.dot(self.W1) + self.B1
        #função de ativação
        self.f1 = np.tanh(self.z1)
        #eq da reta 2
        z2 = self.f1.dot(self.W2) + self.B2
        #função de ativação 2
        f2 = np.tanh(z2)
        #Softmax
        exp_values = np.exp(f2)
        softmax = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return softmax
    def loss(self):
        pass
    def backpropagation(self):
        pass
    def fitr(self):
        pass    
    
    
modelo = NnModel(x,y,hidden_neurons=3,output_neurons=2)
modelo.foward(x)