import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True)
df = data['frame']
plt.hist(df['target'], bins=25)
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.title('Distribution of Target')
plt.show()
sns.heatmap(df.corr().round(2), annot=True)
plt.show()
plt.subplot(1, 2, 1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel('bmi')
plt.ylabel('target')

plt.subplot(1, 2, 2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')
plt.tight_layout()
plt.show()

X = df[['bmi', 's5']]
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.2)
lm = LinearRegression()
lm.fit(x_train, y_train)
y_pred_test = lm.predict(x_test)
y_pred_train = lm.predict(x_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
print("While having bmi + s5")
print(f"Train RMSE = {rmse_train:.2f}, R² = {r2_train:.4f}")
print(f"Test  RMSE = {rmse_test:.2f}, R² = {r2_test:.4f}")

X_ext = df[['bmi', 's5', 'bp']]
x_train, x_test, y_train, y_test = train_test_split(X_ext, y, random_state=5, test_size=0.2)
lm_ext = LinearRegression()
lm_ext.fit(x_train, y_train)
y_pred_test_ext = lm_ext.predict(x_test)
y_pred_train_ext = lm_ext.predict(x_train)
rmse_train_ext = np.sqrt(mean_squared_error(y_train, y_pred_train_ext))
r2_train_ext = r2_score(y_train, y_pred_train_ext)
rmse_test_ext = np.sqrt(mean_squared_error(y_test, y_pred_test_ext))
r2_test_ext = r2_score(y_test, y_pred_test_ext)
print("While having bmi + s5 + bp")
print(f"Train RMSE = {rmse_train_ext:.2f}, R² = {r2_train_ext:.4f}")
print(f"Test  RMSE = {rmse_test_ext:.2f}, R² = {r2_test_ext:.4f}")

"""
a) Which variable would you add next? Why?
I choose bp because it has the second highest correlation 
with the target variable after bmi.Adding bp to the model 
with bmi and s5 improves both training and test performance, 
confirming that bp contributes some extra explanatory power.

b) How does adding it affect the model's performance? 
Training metrics improved as RMSE decreased from 56.56 to 55.33 and 
R² increased from 0.4508 to 0.4745.
Test metrics  also improved as RMSE decreased from 57.18 to to 56.63
and R² increased from 0.4816 to 0.4915.

c) Does it help if you add even more variables?
Adding a few important variables improves the model. 
But after a certain point, adding more (especially weak ones) 
doesn’t help much and can even cause overfitting,
the model fits the training data better but not new data.

"""
