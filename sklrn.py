import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Load data (NumPy arrays)
cal = fetch_california_housing()
X, y = cal.data, cal.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Ridge, Lasso and ElasticNet then OneHotEncoder, OrdinalEncoder. 
ridge = Ridge(alpha=0.1)
lasso = Lasso(alpha=0.1)
eln = ElasticNet(alpha=0.1)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
eln.fit(X_train, y_train)

print(f"\nRidge R2: {ridge.score(X_test, y_test)}")
print(f"Lasso R2: {lasso.score(X_test, y_test)}")
print(f"eln R2: {eln.score(X_test, y_test)}\n")
# Ridge, Lasso and ElasticNet(Which combines both in one). Lasso also called selector. 
