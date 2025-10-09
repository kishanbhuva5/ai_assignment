import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('Auto.csv')
print(df.head())
X = df.drop(columns=['mpg', 'name', 'origin'])
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8]

ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_scores.append(ridge.score(X_test, y_test))

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_scores.append(lasso.score(X_test, y_test))

plt.figure(figsize=(10,6))
plt.plot(alphas, ridge_scores, marker='o', label='Ridge R2')
plt.plot(alphas, lasso_scores, marker='x', label='LASSO R2')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('R2 Score on Test Data')
plt.title('R2 Score vs Alpha for Ridge and LASSO')
plt.legend()
plt.grid(True)
plt.show()

best_ridge_index = ridge_scores.index(max(ridge_scores))
best_ridge_alpha = alphas[best_ridge_index]
best_ridge_score = ridge_scores[best_ridge_index]

best_lasso_index = lasso_scores.index(max(lasso_scores))
best_lasso_alpha = alphas[best_lasso_index]
best_lasso_score = lasso_scores[best_lasso_index]

print(f"Best Ridge alpha: {best_ridge_alpha}, R2 Score: {best_ridge_score:.4f}")
print(f"Best LASSO alpha: {best_lasso_alpha}, R2 Score: {best_lasso_score:.4f}")

"""
Findings:
I used multiple regression to predict car MPG using all numeric features except 'mpg', 'name', and 'origin'. 
Ridge regression generally performs better when predictors are correlated, while LASSO can shrink less important 
features to zero, effectively performing feature selection. By looping over several α values, I examined how 
regularization strength affects R² scores on the test set.
The optimal α for Ridge was 0.1, achieving an R² score of 0.7942, while the optimal α for LASSO was also 0.1, 
with an R² score of 0.7923. Smaller α values produced results close to standard linear regression, while very 
large α values led to underfitting and lower R² scores. Using a log scale for α helped clearly identify the range 
where performance peaked.
"""