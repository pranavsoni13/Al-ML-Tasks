import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# sales data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

# create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# fit linear regression model
model = LinearRegression()
model.fit(X_poly, y)

# predict sales
y_pred = model.predict(X_poly)

# plot results
plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, y_pred, color='red', label='Predicted Sales')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
