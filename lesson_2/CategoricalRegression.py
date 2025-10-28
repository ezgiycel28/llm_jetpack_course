import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

df = sns.load_dataset("tips")

# 0 ve 1 halinde sayısal veriye dönüştür
X_c = df["sex"].map({"Female": 0, "Male": 1})
y_c = df['tip']

model = LinearRegression()
# x versini 2 boyutlu bir diziye çevir
model.fit(X_c.values.reshape(-1, 1), y_c)

y_pred = model.predict(X_c.values.reshape(-1, 1))

print(f"Regresyon Katsayısı (Coefficient): {model.coef_}")
print(f"Intercept (Sabit): {model.intercept_}")

plt.figure(figsize=(10, 6))
plt.scatter(X_c, y_c, label="Gerçek Veriler", alpha=0.5)
plt.plot(X_c, y_pred, color="red", linewidth=2, label="Doğrusal Regresyon")
plt.xlabel("Cinsiyet (0: Female, 1: Male)")
plt.ylabel("Tip ($)")
plt.legend()
plt.title("Sex vs. Tip Regression")
plt.show()