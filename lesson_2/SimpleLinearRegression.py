import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd 

df = sns.load_dataset("tips")
# print(df.head()) #veriyi görmek istersen print'i aç

x = df[["total_bill"]] 
y = df['tip']

#modeli eğit
model = LinearRegression()
model.fit(x, y)

#tahmin et
y_pred = model.predict(x)

#görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Gerçek Veriler (Actual Data)")
plt.plot(x, y_pred, color="red", label="Doğrusal Regresyon (Regression Line)")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.legend()
plt.title("Total Bill vs. Tip Regression")
plt.show()