import sys
import os
working_directory = os.getcwd()
sys.path.insert(0, working_directory)


from sklearn.neural_network import MLPClassifier
from preprocessing.preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ANN:

    def __init__(self):
        self.model = MLPClassifier((35,2),"relu",max_iter=300)

    def train(self,x,y):
        y_shape = y.ravel()
        self.model.fit(x,y_shape)

    def predict(self,x):
        self.model.predict(x)

    def get_params(self):
        return self.model.get_params()

    def getModel(self):
        return self.model

    def score(self,x,y):
        y = y.ravel()
        return self.model.score(x,y)

if __name__ == '__main__':
    X_train, y_train = Preprocessor.access_data_labels(working_directory + "\\datasets\\train_test_split\\train.csv")
    
    classifier = ANN()
    classifier.train(X_train,y_train)

    #classifier = ANN.load("data.txt")
    X_test, y_test = Preprocessor.access_data_labels(working_directory + "\\datasets\\train_test_split\\train.csv")
    predictions = classifier.predict(X_test)
    print("Accuracy is: " + str(classifier.score(X_test,y_test)))