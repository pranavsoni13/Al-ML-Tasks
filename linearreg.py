import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# sales data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 5, 7, 11])  

# create and fit the model
model = LinearRegression()
model.fit(X, y)

# make predictions
y_pred = model.predict(X)   

# plot the data and the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()
