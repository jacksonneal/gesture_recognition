from sys import argv
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns

train = pd.read_csv(argv[1], header=None)
X_train, y_train = train.iloc[:, :-1].values, train.iloc[:, -1].values

test = pd.read_csv(argv[2], header=None)
X_test, y_test = test.iloc[:, :-1].values, test.iloc[:, -1].values

s = StandardScaler()

X_train, X_test = s.fit_transform(X_train), s.fit_transform(X_test)

svc = SVC(kernel='rbf', C=15, gamma=0.01, decision_function_shape='ovr', probability=True)
svc.fit(X_train, y_train)

y_pred_svm = svc.predict(X_test)

print(classification_report(y_test, y_pred_svm))

cm = confusion_matrix(y_test, y_pred_svm)

# Plot CM
f, ax = plt.subplots(figsize=(15, 15))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, linewidths=0.01, cmap="Purples", linecolor="black",
            ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Validation set")
plt.show()
