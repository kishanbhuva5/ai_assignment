import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data=load_diabetes(as_frame=True)
print(data.keys())
print(data.DESCR)
df=data['frame']
print(df)
plt.hist(df['target'], bins=25)
plt.xlabel("target")
plt.show()
sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()
plt.subplot(1,2,1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel("bmi")
plt.ylabel("target")
plt.subplot(1,2,2)
plt.scatter(df['bp'], df['target'])
plt.xlabel("bp")
plt.ylabel("target")
plt.show()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=5)
lr=LinearRegression()
lr.fit(X_train, y_train)
print("Train score:", lr.score(X_train, y_train))
print("Test score:", lr.score(X_test, y_test))
y_pred=lr.predict(X_test)
rmse=((y_test - y_pred)**2).mean()**0.5
print("RMSE:", rmse)
r2=lr.score(X_test, y_test)
print("R2:", r2)
plt.scatter(y_test, y_pred)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.plot([0, 350], [0, 350], '--k')
plt.axis('tight')
plt.show()