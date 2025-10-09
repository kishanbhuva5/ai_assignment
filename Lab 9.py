import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('exams.csv',delimiter=",")
print(df.head())
X=df.iloc[:,0:2]
y=df.iloc[:,-1]
admit_yes = df.loc[y==1]
admit_no=df.loc[y==0]
plt.scatter(admit_yes.iloc[:,0],admit_yes.iloc[:,1],color='green',label='Admit yes')
plt.scatter(admit_no.iloc[:,0],admit_no.iloc[:,1],color='red',label='Admit no')
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.legend()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
print(f"Train Accuracy = {train_accuracy:.4f}")
print(f"Test  Accuracy = {test_accuracy:.4f}")
metrics.ConfusionMatrixDisplay.from_estimator(model,X_test,y_test)
plt.show()
cnf_matrix=metrics.confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
print("Precision:",metrics.precision_score(y_test, y_test_pred))
print("Recall:",metrics.recall_score(y_test, y_test_pred))

y_pred = model.predict(X_test)
X_test_df = X_test.copy()
X_test_df['Actual'] = y_test.values
X_test_df['Predicted'] = y_pred

tp = X_test_df[(X_test_df['Predicted'] == 1) & (X_test_df['Actual'] == 1)]
fp = X_test_df[(X_test_df['Predicted'] == 1) & (X_test_df['Actual'] == 0)]
tn = X_test_df[(X_test_df['Predicted'] == 0) & (X_test_df['Actual'] == 0)]
fn = X_test_df[(X_test_df['Predicted'] == 0) & (X_test_df['Actual'] == 1)]

plt.figure(figsize=(8,6))
plt.scatter(tp.iloc[:, 0], tp.iloc[:, 1], color='green', label='Pred Yes - correct (TP)', marker='+')
plt.scatter(fp.iloc[:, 0], fp.iloc[:, 1], color='green', label='Pred Yes - incorrect (FP)', marker='o')
plt.scatter(tn.iloc[:, 0], tn.iloc[:, 1], color='red', label='Pred No - correct (TN)', marker='+')
plt.scatter(fn.iloc[:, 0], fn.iloc[:, 1], color='red', label='Pred No - incorrect (FN)', marker='o')

plt.xlabel('exam1')
plt.ylabel('exam2')
plt.title('Prediction Results (Correct vs Incorrect)')
plt.legend()
plt.show()


