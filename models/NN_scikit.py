from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from preprocessing.preprocessor import Preprocessor

X_train, y_train = Preprocessor.access_data_labels("..\\datasets\\train_test_split\\train.csv")
X_test, y_test = Preprocessor.access_data_labels("..\\datasets\\train_test_split\\test.csv")

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

hidden = (15) # around 87%
mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print("Hidden Layer(s): " + str(hidden))
print("Accuracy is: " + str(accuracy_score(y_test, predictions)))
print(confusion_matrix(y_test, predictions))


hidden = (30, 15) # 91%
mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print("Hidden Layer(s): " + str(hidden))
print("Accuracy is: " + str(accuracy_score(y_test, predictions)))
print(confusion_matrix(y_test, predictions))


hidden = (45, 30, 10)  # %91
mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=1000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print("Hidden Layer(s): " + str(hidden))
print("Accuracy is: " + str(accuracy_score(y_test, predictions)))
print(confusion_matrix(y_test, predictions))


hidden = (50, 25, 10, 15)  # %91
mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=1000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print("Hidden Layer(s): " + str(hidden))
print("Accuracy is: " + str(accuracy_score(y_test, predictions)))
print(confusion_matrix(y_test, predictions))
