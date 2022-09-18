from symbol import parameters
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class MultipleNeurals:

    def __init__(self, X_train, y_train, X_test, y_test, n1) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.X_train_original = self.X_train
        self.X_test_original = self.X_test

        self.X_train, self.X_test = self.normalize(self.X_train, self.X_test)
        # self.X_train, self.X_test = self.flatten_pixels(self.X_train, self.X_test)

        self.n0 = X_train.shape[0]
        self.n1 = n1
        self.n2 = y_train.shape[0]
        
        self.parameters = self.initialization(self.n0, self.n1, self.n2)

        self.trained = False

    @staticmethod
    def normalize(train_set, test_set):
        return (train_set / train_set.max(), test_set / test_set.max())

    @staticmethod
    def flatten_pixels(train_set, test_set):
        train_set = train_set.reshape(train_set.shape[0], train_set.shape[1]*train_set.shape[2])
        test_set = test_set.reshape(test_set.shape[0], test_set.shape[1]*test_set.shape[2])
        return (train_set, test_set)


    @staticmethod
    def initialization(n0, n1, n2):
        """
        n0 : network input number
        n1 : first layer neurals number
        n2 : second layer neurals number

        """
        W1 = np.random.randn(n1, n0)
        b1 = np.random.randn(n1, 1)
        W2 = np.random.randn(n2, n1)
        b2 = np.random.randn(n2, 1)

        return { 'W1' : W1,
                 'b1' : b1,
                 'W2' : W2,
                 'b2' : b2 }

    @staticmethod
    def forward_propagation(X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        Z1 = W1.dot(X) + b1
        A1 = 1 / (1 + np.exp(-Z1))
        Z2 = W2.dot(A1) + b2
        A2 = 1 / (1 + np.exp(-Z2))
        
        return { 'A1' : A1,
                 'A2' : A2 }

    @staticmethod
    def log_loss(A, y, epsilon = 1e-15):
        return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1-y) * np.log(1 - A + epsilon))

    @staticmethod
    def back_propagation(X, y, activations, parameters):

        A1 = activations['A1']
        A2 = activations['A2']
        W2 = parameters['W2']

        m = y.shape[1]

        dZ2 = A2 - y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        
        dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        return { 'dW1' : dW1,
                 'db1' : db1,
                 'dW2' : dW2,
                 'db2' : db2 }

    @staticmethod
    def update(gradients, parameters, learning_rate):

        dW1 = gradients['dW1']
        db1 = gradients['db1']
        dW2 = gradients['dW2']
        db2 = gradients['db2']

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        return { 'W1' : W1,
                 'b1' : b1,
                 'W2' : W2,
                 'b2' : b2 }

    def train(self, learning_rate = 0.02, n_iter = 5000):
        if self.trained:
            print("The model has been already trained.")
            return
            
        train_loss = []
        history = []

        for i in tqdm(range(n_iter)):
            activations = self.forward_propagation(self.X_train, self.parameters)
            gradients = self.back_propagation(self.X_train, self.y_train, activations, self.parameters)
            self.parameters = self.update(gradients, self.parameters, learning_rate)

            if i % 10 == 0:
                train_loss.append(self.log_loss(activations["A2"], self.y_train))
                history.append([self.parameters, train_loss[int(i/10)]])

        # plt.plot(Loss)
        # plt.show()
        y_pred = self.predict(self.X_train, self.parameters)
        print("Accuracy score: ", accuracy_score(self.y_train.flatten(), y_pred.flatten()))

        self.trained = True

    def predict(self, X, parameters):
        activations = self.forward_propagation(X, parameters)
        A2 = activations["A2"]
        return A2 >= 0.5

    def show_train_set(self):
        plt.figure(figsize=(16,8))
        for i in range(1,15):
            plt.subplot(4, 5, i)
            plt.imshow(self.X_train_original[i], cmap='gray')
            plt.title("chien" if self.y_train[i] == 1.0 else "chat")
            plt.tight_layout()
        plt.show()

    def show_test_set(self):
        y_predict = self.predict(self.X_train, self.parameters)
        for i in range(1,15):
            plt.subplot(4, 5, i)
            plt.imshow(self.X_train_original[i], cmap='gray')
            plt.title("chien" if y_predict[i] == 1.0 else "chat")
            plt.tight_layout()
        plt.show()