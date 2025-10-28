import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler # SVR için ölçekleme şarttır
import pandas as pd
import numpy as np

df = sns.load_dataset("tips")
x_df = df[["total_bill"]]
y_df = df[["tip"]] # y'yi de ölçeklemek için 2D yap

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x_df)
y_scaled = scaler_y.fit_transform(y_df)
y_scaled_flat = y_scaled.flatten() # modelin istediği 1D formatı haline getir

model = SVR(kernel="linear")#linear-->düz çizgi bulmaya çalış
model.fit(x_scaled, y_scaled_flat)

y_pred_scaled = model.predict(x_scaled)

y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

r2_skoru = model.score(x_scaled, y_scaled_flat)
print(f"SVR Modelinin R-Kare (R2) Skoru: {r2_skoru:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(x_df, y_df, color='blue', label='Gerçek Veriler (Actual)', alpha=0.5)
plt.plot(x_df, y_pred_unscaled, color="red", linewidth=2, label="SVR Regresyon Çizgisi")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.legend()
plt.title("Support Vector Regression (SVR)")
plt.show()