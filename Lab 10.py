import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('iris.csv',delimiter=",")
print(df.head())
X=df.iloc[:,0:4]
y=df.iloc[:,-1]


classfier = KNeighborsClassifier(n_neighbors=5)
classfier.fit(X, y)
y_pred = classfier.predict(X)
metrics.ConfusionMatrixDisplay.from_estimator(classfier,X,y)
plt.show()
metrics.classification_report(y, y_pred, output_dict=True)
print(classification_report(y, y_pred))
