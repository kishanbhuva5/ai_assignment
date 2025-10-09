import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df=pd.read_csv('bank.csv',delimiter=";")
print(df.head())
print(df.dtypes)
print(df.shape)
print(list(df.columns))

cols = ['y', 'job', 'marital', 'default', 'housing', 'poutcome']
df2 = df[cols].copy()
df3 = pd.get_dummies(df2, columns=['job','marital','default','housing','poutcome'], drop_first=True)

if df3['y'].dtype == 'object':
    df3['y'] = df3['y'].map({'no':0, 'yes':1})
corr = df3.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=False)
plt.title("Correlation matrix for df3")
plt.tight_layout()
plt.show()

corr_with_y = corr['y'].drop('y').abs().sort_values(ascending=False)
print("\nTop 10 Absolute correlations with target 'y':")
print(corr_with_y.head(10))

y=df3['y']
X=df3.drop('y',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)
print("\nLogistic Regression:")
print("Confusion matrix:\n", cm_lr)
print("Accuracy:", acc_lr)

display1 = metrics.ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test)
plt.show()
print(classification_report(y_test, y_pred_lr))
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
print("K-Nearest Neighbors (k=3)")
print("Confusion matrix:\n", cm_knn)
print("Accuracy:", acc_knn)

display2 = metrics.ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
plt.show()
print(classification_report(y_test, y_pred_knn))

"""
Top 10 Absolute correlations with target 'y':
poutcome_success    0.283481
poutcome_unknown    0.162038
housing_yes         0.104683
job_retired         0.086675
job_blue-collar     0.068147
marital_married     0.064643
poutcome_other      0.051908
job_student         0.047809
marital_single      0.045815
job_management      0.032634

Overall, the amount of correlation is generally low, suggesting that the predictive 
power likely comes from combining multiple features rather than relying on a single highly correlated variable.

Results comparison:
Logistic Regression:
Confusion matrix:
    [[997   9]
    [  106  19]]
Accuracy: 0.8983
K-Nearest Neighbors (k=3)
Confusion matrix:
    [[968   38]
    [ 108  17]]
Accuracy: 0.8709

Logistic Regression achieved an accuracy of 89.8%, with strong performance on the 
negative class (precision = 0.90, recall = 0.99) but low recall (0.15) for the positive class.
K-Nearest Neighbors (k = 3) reached a lower accuracy of 87.1%, with more false positives and 
weaker precision (0.31) and recall (0.14) for the positive class.

Logistic Regression outperforms KNN on this dataset, particularly in accuracy and precision. 
However, both models have low recall for the minority class, indicating a need for strategies to 
address class imbalance if correctly identifying the positive class is important.
"""
