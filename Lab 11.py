import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df=pd.read_csv('iris.csv',delimiter=",")
print(df.head())
X=df.drop('species',axis=1)
y=df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
svclassifier=SVC(kernel='poly',degree=2)
svclassifier.fit(X_train,y_train)
y_pred=svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))