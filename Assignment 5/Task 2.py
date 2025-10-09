import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("50_Startups.csv")
print("First 5 rows of the dataset:")
print(df.head())
print(df.info())
numeric_df = df.select_dtypes(include=[np.number])
print("\nCorrelation matrix:")
print(numeric_df.corr())

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr().round(2), annot=True)
plt.title('Correlation between numeric variables')
plt.show()

df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
X = df_encoded.drop('Profit', axis=1)
y = df_encoded['Profit']

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(df['R&D Spend'], df['Profit'])
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('R&D Spend vs Profit')

plt.subplot(1,2,2)
plt.scatter(df['Marketing Spend'], df['Profit'])
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.title('Marketing Spend vs Profit')
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Train RMSE = {train_rmse:.2f}, R² = {train_r2:.4f}")
print(f"Test  RMSE = {test_rmse:.2f}, R² = {test_r2:.4f}")

"""
Findings:
The dataset has 50 entries and 5 columns: R&D Spend, Administration, Marketing Spend, State, and Profit.
Profit is strongly correlated with R&D Spend (i.e 0.97) and 0.75 with Marketing Spend.
Administration has weak correlation with Profit (i.e. 0.20).
A regression model trained on this data achieved a training RMSE of 8927.49 and an R² of 0.9537, 
and a test RMSE of 9055.96 with an R² of 0.8987, indicating the model explains most of the variation 
in Profit and generalizes well with minimal overfitting. These results suggest that R&D Spend is 
the primary driver of Profit, Marketing Spend also contributes significantly, 
and Administration has minimal impact on predictive performance.
"""
