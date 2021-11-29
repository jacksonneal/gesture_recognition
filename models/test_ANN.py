from random import seed,shuffle
from csv import reader


from models.ANN import ArtificialNeuralNetwork



# loading the file
def load_csv(self,filename):
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
def convert_to_float(self,data, column):
    for row in data:
        row[column] = float(row[column].strip())
#converts colum to integer
def convert_to_integer(self,data, column):
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
def minmax(self,dataset):
    maximum = [max(i) for i in zip(*dataset)]
    minimum = [min(i) for i in zip(*dataset)]
    return maximum,minimum

# Normalize the dataset
def normalize(self,dataset, minmax=minmax):

    colMax, colMin = minmax(dataset)
    denom = [a-b for a,b in zip(colMax,colMin)]
    dataNorm = dataset.copy()
    #Min-max scaling (X-Xmin)/(Xmax-Xmin)
    #loop over rows and caluculate normalized
    for i in dataNorm:
        for j in range(len(i)-1):
            i[j] = (i[j]-colMin[j]) / (denom[j])
    return dataNorm

# Split a dataset into k folds
def cross_validation_split(self,dataset, n_folds):
    # #Get the length of each fold
    length = int(len(dataset)/n_folds)
    folds = []
    #Add to fold
    for i in range(n_folds-1):
        folds += [dataset[i*length:(i+1)*length]]
    #Get the remaining into a fold
    folds += [self.data[(n_folds-1)*length:len(dataset)]]
    
    return folds




def maine():

    seed(1)
    # load and prepare data
    data = load_csv('test.csv')
    data = data[1:]

    #shuffle data
    shuffle(data)
    for i in range(len(data[0])-1):
        convert_to_float(data, i)
    convert_to_integer(data,-1)

    # normalize input variables
    dataset = normalize(data)

    # evaluate algorithm
    n_folds = 5
    l_rate = 0.3
    n_epoch = 500
    n_hidden = 4

    model = ArtificialNeuralNetwork(dataset,n_folds,l_rate,n_epoch,n_hidden)

    count = model.evaluate_algorithm(dataset, n_folds, l_rate, n_epoch, n_hidden)
    print('Scores: %s' % count)
    print('Mean Accuracy: %.3f%%' % (sum(count)/float(len(count))))

if __name__ == "__main__":
    main()