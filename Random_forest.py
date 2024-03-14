# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


# Importing the dataset
dataset = pd.read_csv('GE.csv')
X = dataset['open'].values
y = dataset['close'].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)
X = np.reshape(X,(len(X),1))
y = np.reshape(y, (len(y), 1))
regressor.fit(X, y)

# Predicting a new result
# y_pred = regressor.predict(6.5)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Opening stock price')
plt.ylabel('Closing Stock price')
plt.show()

#Calculate accuracy 

# After fitting the model (regressor.fit(X, y))
y_pred = regressor.predict(X)  # Predict closing prices for all data
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)