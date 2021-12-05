from random import seed, random, randrange, shuffle
from csv import reader
from math import exp
import sys
from IPython.core.debugger import set_trace

class ArtificialNeuralNetwork:

    def __init__(self,dataset, n_folds, l_rate, n_epoch, n_hidden):
        self.dataset = dataset
        self.n_folds = n_folds
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.n_hidden = n_hidden

        self.network = None
        self.dropout = []
        

    

    ####################################################################################

    # Initialize a network
    def initialize_network(self,n_inputs, n_hidden, n_outputs):
        network = []
        #Create hidden layer
        hiddenLayer = [{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
        #Add to the array
        network.append(hiddenLayer)
        
        #create ouput and save to list
        outputLayer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
        network.append(outputLayer)
        

        return network


    # Split a dataset into k folds
    def cross_validation_split(self,dataset, n_folds):
        # #Get the length of each fold
        length = int(len(dataset)/n_folds)
        folds = []
        #Add to fold
        for i in range(n_folds-1):
            folds += [dataset[i*length:(i+1)*length]]
        #Get the remaining into a fold
        folds += [dataset[(n_folds-1)*length:len(dataset)]]
        
        return folds


    # Calculate neuron activation for an input
    def activate(self,weights, inputs):
        #let bias be the last weight in the weights, i.e. Wn,Wn-1,...W1, b
        #such that there are one more weight than input to node

        activationValue = 0
        #Sum the weights and inputs
        for i in range(len(weights)-1 ):
            activationValue += weights[i] * inputs[i]
        #Add the bias term
        activationValue += weights[-1]	
        return activationValue


    # Transfer neuron activation
    def transfer(self,activation):
        #Use Sigmoid for now, add Relu for project later
        if activation >=0:
            return 1. / (1. + exp(-activation))
        else:
            return exp(activation) / (1. + exp(activation))


    # Forward propagate input to a network output
    def forward_propagate(self,network, row):

        inputs = row
        #Go through layer in network
        for i in network:
            outPuts = []
            #get nodes in each layer
            for node in i:
                #Calculate the activation and transfer of each node and append to a outputs list
                activation = self.activate(node['weights'],inputs)
                node['output'] = self.transfer(activation)
                outPuts.append(node['output'])
            #save the outputs of the nodes in layer
            #At end input = values at output layer
            inputs = outPuts

        return inputs

    # Calculate the derivative of an neuron output
    def transfer_derivative(self,output):
        return output * (1.0-output)

    # Backpropagate error and store in neurons
    def backward_propagate_error(self,network, expected):
        #Loop from output back
        for i in reversed(range(len(network))):
            #get layer and make an error list
            layer = network[i]
            errors = []
            #If i is not at end
            if i != len(network)-1:
                for k in range(len(layer)):
                    error = 0.0
                    #Look at nodes in next layer
                    for node in network[i+1]:
                        #error for kth node is sum from node in next layer delta = error_j*transfer derivative 
                        error += (node["weights"][k] * node['delta'])
                    errors.append(error)
            else:
                #Output layer
                #Loop through all the nodes storing the error
                for j in range(len(layer)):
                    node = layer[j]
                    errors.append(node['output']- expected[j])
            #calculate error_j * transferderivative for layer
            for j in range(len(layer)):
                node = layer[j]
                node['delta'] = errors[j] * self.transfer_derivative(node['output'])

    # Backpropagation Algorithm With Stochastic Gradient Descent
    def back_propagation(self,train, test, l_rate, n_epoch, n_hidden):
        #Get the number of input variables
        numInputs = len(train[0])-1
        #get number of outputs
        numOutputs = len(set([i[-1] for i in train]))
        #Create a network
        network = self.initialize_network(numInputs,n_hidden,numOutputs)
        #Train network with values
        self.train_network(network,train,l_rate,n_epoch,numOutputs)

        results = []
        for row in test:
            results.append( self.predict(network,row))
        return results


    # Update network weights with error
    def update_weights(self,network, row, l_rate):
        
        #Loop through rows to update weighs
        for i in range(len(network)):
            #Grab inputs
            inputs = row[:-1]
            #if not first layer
            if i != 0:
                #set input to the outputs of the previous layer
                inputs = [node['output'] for node in network[i-1]]
            for node in network[i]:
                for j in range(len(inputs)):
                    #subtract learning rate * [error * input] <-(delta)
                    node['weights'][j] -= l_rate * node['delta'] * inputs[j]
                #
                node['weights'][-1] -= l_rate * node['delta']

    # Train a network for a fixed number of epochs
    def train_network(self,network, train, l_rate, n_epoch, n_outputs):
        #loop through number of epochs
        for i in range(n_epoch):
            
            for row in train:
                #get outputs by propogating the row in network
                outputs = self.forward_propagate(network, row)
                
                #find the expected for each output
                expected = [0 for i in range(n_outputs)]
                #update the expected[label]
                expected[row[-1]] = 1
                #back propagate expected through network
                self.backward_propagate_error(network, expected)
                #
                self.update_weights(network, row, l_rate)

                

                
            


    # Make a prediction with a network
    def predict(self,network, row):
        outputs = self.forward_propagate(network, row)
        #return the index of output neuron with highest value
        return outputs.index(max(outputs))

    #new function to 


    # Calculate accuracy percentage
    def accuracy_metric(self,actual, predicted):
        #correctly identified
        correct = 0
        #total amount of predicted
        total = len(predicted)

        for i in range(total):
            #increment correct if actual = predicted
            if actual[i] == predicted[i]:
                correct += 1

        #print(total)
        #print(correct)
            
        return 100 * (correct / float(total))


    #Find accuracy of each label
    def label_accuracy(self,actual, predicted):
        total = len(predicted)
        errorCounts = [0 for i in range(4)]
        counts = [0 for i in range(4)]

        for i in range(total):
            
            if actual[i] == 0:
                counts[0] += 1
            elif actual[i] == 1:
                counts[1] += 1
            elif actual[i] ==2:
                counts[2] += 1
            else:
                counts[3] += 1
            
            #count correct
            if actual[i] == predicted[i]:
                if actual[i] == 0:
                    errorCounts[0] += 1
                elif actual[i] ==1:
                    errorCounts[1] += 1
                elif actual[i] ==2:
                    errorCounts[2] += 1
                else:
                    errorCounts[3] += 1

        #print('correct label 0: %s' %counts[0])
        #print('correct count: %s' %errorCounts[0])

        return [100*(a / float(b)) for a,b in zip(errorCounts,counts)]



    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self,dataset,  n_folds, *args):
        #get folds from data
        folds = self.cross_validation_split(dataset,n_folds)
        #Loop over each fold to remove
        accuracy = []
        labelAcc = []

        for fold in folds:
            #make new list of folds
            newFolds = list(folds)
            newFolds.remove(fold)
            train = newFolds
            
            
            #merge back
            train = sum(train,[])
            #Create the test lsit
            test = list()
            for i in fold:
                #Copy row and add to test
                copyRow = list(i)
                #remove the label
                copyRow[-1] = None
                test.append(copyRow)
            #make a prediction
            prediction = self.back_propagation(train,test,*args)
            actual = [row[-1] for row in fold]

            print(prediction)
            
            
            #calculate accuracy and save to list
            labelAcc.append(self.label_accuracy(actual,prediction))
            accuracy.append(self.accuracy_metric(actual,prediction))
                
                
            #Look into train



        return accuracy,labelAcc





##################################################################################

# loading the file
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Preprocessing the data to fit your implementation
#converts column to float
def convert_to_float(data, column):
    for row in data:
        row[column] = float(row[column].strip())
#converts colum to integer
def convert_to_integer(data, column):
    #map labels to natural numbers sequentially
    labels = [row[column] for row in data]
    #get the unique elements of the labels
    uniqueLabels = set(labels)
    #create a dict to map(convert) label into unique code
    converter = dict()
    #loop through unique labels assigning a unique number starting at 0
    for i,label in enumerate(uniqueLabels):
        converter[label] = i
    #convert the data
    for row in data:
        row[column] = converter[row[column]]

# Find the min and max values for each column
def minmax(dataset):
    maximum = [max(i) for i in zip(*dataset)]
    minimum = [min(i) for i in zip(*dataset)]
    return maximum,minimum

# Normalize the dataset
def normalize(dataset, minmax=minmax):

    colMax, colMin = minmax(dataset)
    denom = [a-b for a,b in zip(colMax,colMin)]
    dataNorm = dataset.copy()
    #Min-max scaling (X-Xmin)/(Xmax-Xmin)
    #loop over rows and caluculate normalized
    for i in dataNorm:
        for j in range(len(i)-1):
            i[j] = (i[j]-colMin[j]) / (denom[j])
    return dataNorm





#main funciton to produce model analysis
def main():
    import os
    from sklearn.neural_network import MLPClassifier
    import numpy as np
    from sklearn.model_selection import train_test_split


    seed(1)
    # load and prepare data
    working_directory = os.getcwd()
    file_path = working_directory + '\datasets\\train_test_split\\train.csv'
    #file_path = working_directory + '\seeds.csv'
    
    data = load_csv(file_path)
    dataset = data[1:]

    #shuffle data
    shuffle(data)

    data = dataset[0:500]
    #dataTest = dataset[500:600]

    #convert data
    for i in range(len(data[0])-1):
        convert_to_float(data, i)
    convert_to_integer(data,-1)

    # normalize input variables
    dataset = normalize(data)

    # evaluate algorithm
    n_folds = 5
    l_rate = 0.1
    n_epoch = 50
    n_hidden = 35
    
   


    model = ArtificialNeuralNetwork(dataset,n_folds,l_rate,n_epoch,n_hidden)
    
    
    
    count,labelCount = model.evaluate_algorithm(dataset, n_folds, l_rate,n_epoch, n_hidden)

    


    print('Scores: %s' % count)
    print('Label scores: %s' % labelCount)
    print('Mean Accuracy: %.3f%%' % (sum(count)/float(len(count))))

if __name__ == "__main__":
    main()
       
