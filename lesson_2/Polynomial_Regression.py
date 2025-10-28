import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures # Yazım hatası
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np 

df = sns.load_dataset("tips")
x = df[["total_bill"]]
y = df['tip']

poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(X_poly, y)

x_sorted = np.sort(x, axis=0) 
X_poly_sorted = poly.transform(x_sorted) 
y_pred_sorted = model.predict(X_poly_sorted) 

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Gerçek Veriler", alpha=0.5)
plt.plot(x_sorted, y_pred_sorted, color="red", label="Polinomsal Regresyon (Derece 3)") 
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.legend()
plt.title("Polynomial Regression")
plt.show()