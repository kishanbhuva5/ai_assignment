import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('weight-height.csv')
plt.scatter(data['Height'], data['Weight'], alpha=0.2)
plt.xlabel('Height in inches')
plt.ylabel('Weight in pounds')
plt.title('Scatter plot of Height vs Weight')
plt.show()
X = data[['Height']].values
y = data['Weight'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(X, y, alpha=0.3, label='Actual data')
plt.plot(X, y_pred, color='green', label='Regression line')
plt.xlabel('Height in inches')
plt.ylabel('Weight in pounds')
plt.title('Linear Regression of Height vs Weight')
plt.legend()
plt.show()
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE: {rmse:.2f}")
r2 = r2_score(y, y_pred)
print(f"R²: {r2:.4f}")
