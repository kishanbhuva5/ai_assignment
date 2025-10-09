import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('weight-height.csv')
plt.scatter(data['Height'], data['Weight'], alpha=0.2)
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.title('Scatter plot of Height & Weight')
plt.show()
X = data[['Height']].values
y = data['Weight'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(X, y, alpha=0.3, label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.title('Linear Regression of Height & Weight')
plt.legend()
plt.show()
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE: {rmse:.2f}")
r2 = r2_score(y, y_pred)
print(f"R²: {r2:.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
rmse_test = mean_squared_error(y_test, y_test_pred)
print(f"Test RMSE: {rmse_test:.2f}")
r2_test = r2_score(y_test, y_test_pred)
print(f"Test R²: {r2_test:.4f}")

