import sys
import os
import json


working_directory = os.getcwd()
sys.path.insert(0, working_directory)


from sklearn.neural_network import MLPClassifier
from preprocessing.preprocessor import Preprocessor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ANN:

    def __init__(self,layout=(75,)):
        self.layout = layout
        self.model = 'adam'
        self.model = MLPClassifier(layout,"relu",max_iter=500)

    def train(self,x,y):
        #model needs 1d array not column vector
        y_shape = y.ravel()
        self.model.fit(x,y_shape)

    def predict(self,x):
        return self.model.predict(x)

    def get_params(self):
        return self.model.get_params()

    def getModel(self):
        return self.model

    def setModel(self, layout, model = 'adam'):
        self.layout = layout
        self.model = model
        self.model = MLPClassifier(layout,solver=model,max_iter=500)

    def score(self,x,y):
        #model needs 1d array not column vector
        y = y.ravel()
        return self.model.score(x,y)

    def triple_Score(self,predict,y_test):
        x = predict.ravel()
        y = y_test.ravel()
        scores = [0,0,0]
        scores[0] = accuracy_score(x,y)
        #Average is set to macro which is produces a non weighted average
        #Since classes are approximately balanced it is not needed
        scores[1] = precision_score(x,y,average='macro')
        scores[2] = recall_score(x,y,average='macro')
        return scores

    def confusion(self, predicted,actual):
        ConfusionMatrixDisplay.from_predictions(actual,predicted)
        plt.title("NN Confusion Matrix")
        plt.show()
        

    


    def test_model(self,X_train,y_train,X_test,y_test,display = False):
         #Produce matrix of training accuracy, precision, and recall over #layers, #nodes, #iterations?
        bestScore = 0
        modelParams = (0)
        results = []
        for layer in range(1,6):
            scores = []
            for nodes in range(20,90,10):
                #create new classifier
                layout = []

                for i in range(layer):
                    layout.append(nodes)
                
                layout = tuple(layout)

                self.setModel(layout)
                self.train(X_train,y_train)
                score = self.score(X_test,y_test)
                scores.append(score)
                
                #save best score
                if score > bestScore:
                    bestScore = score
                    modelParams = layout
                if display:
                    print("Accuracy for model with layout %s is: %s" % (layout ,score))
            results.append(scores)
        
        self.setModel(modelParams)
        self.train(X_test,y_test)

        return results

    def test_Overfitting(self,X_train,y_train,X_test,y_test):
       
        trainResults = []
        testResults = []
        nodes = 15

        for layer in range(1,6):
            model = ANN()
            #create new classifier
            layout = []
            #Create tuple of nodes
            for i in range(layer):
                layout.append(nodes)
            layout = tuple(layout)
            #
            model = ANN(layout,)
            model.train(X_train,y_train)
            score1 = model.score(X_train,y_train)
            score2 = model.score(X_test,y_test)
            trainResults.append(score1)
            testResults.append(score2)
            

        plt.plot(trainResults, label = 'Training')
        plt.plot(testResults, label = 'Test')
        plt.title("Accuracy vs Number of Layer")
        plt.xlabel("Number of layers")
        plt.ylabel("Accuracy")
        plt.legend() 
        plt.show()
        
        

    #function that test solvers
    def test_Solver(self,display = False):

        solvers = ['lbfgs','sgd','adam']
        for i in solvers:
        
            self.setModel(self.layout,i)
            self.train(X_train,y_train)
            score = self.score(X_test,y_test)
            if display:
                print('The score for %s solver is: %s' %(i,score))

    #produce metrics for a trained 
    def acc_By_Label(self,X_train,y_train,X_test,y_test):
         
        #Produce acuracy of each label
        count = [0 for i in range(4)]
        correct = [0 for i in range(4)]

        predictions = self.predict(X_test)
        
        for i in range(len(predictions)):
            if predictions[i] == 0:
                count[0] = count[0] + 1
                if predictions[i]==y_test[i]:
                    correct[0] =correct[0] + 1
            if predictions[i] == 1:
                count[1] =count[1] + 1
                if predictions[i]==y_test[i]:
                    correct[1] += 1
            if predictions[i] == 2:
                count[2] = count[0] + 1
                if predictions[i]==y_test[i]:
                    correct[2] += 1
            if predictions[i] == 0:
                count[3] += 1
                if predictions[i]==y_test[i]:
                    correct[3] += 1
        accuracyByLabel = [100* (a / float(b)) for a, b in zip(correct,count)]
        
        return accuracyByLabel


    
    def plot_Training(self, scores):
        for i in range(len(scores)):
            plt.plot(scores[i],label = "%s layers" %(i+1))

        plt.title("Accuracy vs Nodes in layer")
        plt.xlabel("Number of nodes x10")
        plt.ylabel("Accuracy")  
        plt.legend()
        plt.show()

    def plot_Compare_Layers(self):
        #Compare learning curve of one hidden layer to 4
        class1 = ANN((60))
        class2 = ANN((60,60,60,60))

        class1.train(X_train,y_train)
        class2.train(X_train,y_train)
        score1 = class1.score(X_test,y_test)
        score2 = class2.score(X_test,y_test)
        curve1 =class1.getModel().loss_curve_
        curve2 = class2.getModel().loss_curve_


        plt.plot(curve1,label="1 hidden layer")
        plt.plot(curve2,label = "4 hidden layers")
        plt.title("Training error, 1 hidden layer vs 4")
        plt.xlabel("n iterations")
        plt.ylabel("error")  
        plt.legend() 
        plt.show()


        print("The scores for 1 layer and 4 are %s,%s respectivley" % (score1,score2))
        
        


    #load and save not working 
    def save(self, dest):
        obj = self.model
        with open(dest, "w") as f:
            json.dump(obj, f, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @staticmethod
    def load(src):
        loaded_classifier = ANN()
        loaded_classifier.training_data_len = 0
        loaded_classifier.model = {}

        with open(src, "r") as f:
            obj = json.load(f)

        for key in obj:
            loaded_classifier.model[float(key)] = obj[key]
            loaded_classifier.training_data_len += obj[key][0][2]

        return loaded_classifier




if __name__ == '__main__':
    X_train, y_train = Preprocessor.access_data_labels(working_directory + "\\datasets\\train_test_split\\train.csv")
    X_test, y_test = Preprocessor.access_data_labels(working_directory + "\\datasets\\train_test_split\\test.csv") 
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    classifier = ANN()

    
    #loops over parameters
    output = classifier.test_model(X_train,y_train,X_test,y_test,display=False)
    classifier.test_Solver(True)

    #graph each row of matrix
    classifier.plot_Training(output)

    #Graph overfitting
    classifier.test_Overfitting(X_train,y_train,X_test,y_test)

    #predict with found parameters
    predict = classifier.predict(X_test)

    classifier.plot_Compare_Layers()
    
    #produce confusion
    print('chosen layout is: {}'.format(classifier.layout))
    #The overall accuracy, precision and recall
    print("Overall accuracy, precision, and recall is: %s" %classifier.triple_Score(predict,y_test))
    classifier.confusion(predict,y_test)
    #print by label
    accuracyByLabel = classifier.acc_By_Label(X_train,y_train,X_test,y_test)
    print('Overall Accuracy is: %s' % classifier.score(X_test,y_test))
    print('The accuracy by label is: %s' % accuracyByLabel)
    #K fold
    print("Cross fold validation: %s" % cross_val_score(classifier.getModel(),X_train,y_train))