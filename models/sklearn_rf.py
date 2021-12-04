from sys import argv

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

train = pd.read_csv(argv[1], header=None)
X_train, y_train = train.iloc[:, :-1].values, train.iloc[:, -1].values

test = pd.read_csv(argv[2], header=None)
X_test, y_test = test.iloc[:, :-1].values, test.iloc[:, -1].values

s = StandardScaler()

X_train, X_test = s.fit_transform(X_train), s.fit_transform(X_test)

RF = RandomForestClassifier(oob_score=True,
                            random_state=42,
                            warm_start=True,
                            n_jobs=-1)

oob_list = []
for n_trees in [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]:
    RF.set_params(n_estimators=n_trees)
    RF.fit(X_train, y_train)
    oob_error = 1 - RF.oob_score_
    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))

rf_oob_df = pd.concat(oob_list, axis=1).T.set_index('n_trees')

sns.set_context('paper')
sns.set_style('white')

ax = rf_oob_df.plot(legend=False, marker='o', figsize=(14, 7), linewidth=5)
ax.set(ylabel='out-of-bag error')

RF_300 = RandomForestClassifier(n_estimators=300
                                , oob_score=True
                                , random_state=42
                                , n_jobs=-1)

RF_300.fit(X_train, y_train)
oob_error300 = 1 - RF_300.oob_score_

y_pred_rf = RF_300.predict(X_test)
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)

# Plot CM
f, ax = plt.subplots(figsize=(15, 15))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, linewidths=0.01, cmap="Purples", linecolor="black",
            ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Validation set")
plt.show()
