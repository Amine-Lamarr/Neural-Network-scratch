import numpy as np

class aminecanfly:
    def __init__(self):
        self.w1 = np.random.randn(10, 6)     
        self.w2 = np.random.randn(6 , 4)     
        self.w3 = np.random.randn(4 , 1)     
        self.b1 = np.random.randn(1 , 6)
        self.b2 = np.random.randn(1 , 4)
        self.b3 = np.random.randn(1 , 1)   
        self.err = []  
        self.epochs = 0

    def logloss(self , pred , y):
        epsilon = 1e-15
        pred = np.clip(pred , epsilon , 1-epsilon)
        loss = - np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        return loss
 
    def relu(self , z):
        return np.maximum(0 , z)

    def dervrelu(self, z):
        return (z > 0).astype(int) 
    
    def sigmoid(self , z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self , x, y, epochs , learning_rate):
        self.epochs = epochs
        for i in range(epochs):
            # forward pass
            z1 = np.dot(x , self.w1) + self.b1         
            a1 = self.relu(z1)

            z2 = np.dot(a1 , self.w2) + self.b2        
            a2 = self.relu(z2)

            z3 = np.dot(a2 , self.w3) + self.b3        
            a3 = self.sigmoid(z3)

            # backpropagation
            dz3 = a3 - y                               
            dw3 = np.dot(a2.T , dz3)                   
            db3 = np.sum(dz3 , axis=0 , keepdims=True) 

            da2 = np.dot(dz3 , self.w3.T)              
            dz2 = da2 * self.dervrelu(z2)               
            dw2 = np.dot(a1.T , dz2)                    
            db2 = np.sum(dz2 , axis= 0 , keepdims=True) 

            da1 = np.dot(dz2 , self.w2.T)               
            dz1 = da1 * self.dervrelu(z1)               
            dw1 = np.dot(x.T , dz1)                     
            db1 = np.sum(dz1 , axis= 0 , keepdims=True) 

            # update weights
            self.w3 -= learning_rate * dw3
            self.w2 -= learning_rate * dw2
            self.w1 -= learning_rate * dw1
            self.b1 -= learning_rate * db1
            self.b2 -= learning_rate * db2
            self.b3 -= learning_rate * db3

            # compute loss
            loss = self.logloss(a3 , y)
            self.err.append(loss)
            if i % 50 == 0 :
                print(f"Epoch {i} | Loss: {loss:.2f}")
    
    def predict(self , x_test):
        z1 = np.dot(x_test , self.w1) + self.b1
        a1 = self.relu(z1)

        z2 = np.dot(a1 , self.w2) + self.b2
        a2 = self.relu(z2)

        z3 = np.dot(a2 , self.w3) + self.b3
        a3 = self.sigmoid(z3)

        return (a3 > 0.5).astype(int)
    
    def accuracy(self, pred, y_true):
        return np.mean(pred == y_true)
    

# let's give it a small test 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , recall_score


x = np.random.randn(200, 10)
y = (np.random.rand(200, 1) > 0.5).astype(int)
x_train , x_test , y_train , y_test = train_test_split(x ,y,test_size=0.2 , random_state=23)

# model 
model = aminecanfly()
model.fit(x_train, y_train, epochs=300, learning_rate=0.01)

# predictions
preds = model.predict(x_test)
cm = confusion_matrix(y_test , preds)
recall = recall_score(y_test, preds)

print(f"recall score : {recall*100:.2f}%")
print(f"Accuracy : {model.accuracy(preds , y_test)*100:.2f}%")

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm , annot=True , fmt='d' , cmap='viridis')
plt.xlabel("x")
plt.ylabel("y")
plt.title("aminecanfly's model")
plt.show()

plt.plot(range(model.epochs) , model.err , label = 'cost function' , c = 'purple', marker = 'x' , ms = 10 )
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('loss variation')
plt.show()