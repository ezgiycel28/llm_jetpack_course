#---------------DENETİMSİZ ÖĞRNEME / KMEANS MODELİ-----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from sklearn import datasets


iris = datasets.load_iris()
x = iris.data
y_true = iris.target 

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')#verilerde 3 adet grup olduğunun varsayıyorum
y_pred = kmeans.fit_predict(x_scaled)#modele gerçek verileri vermiyoruz

df = pd.DataFrame(x, columns=iris.feature_names)
df['cluster (Tahmin Edilen)'] = y_pred 
df['target (Gerçek)'] = y_true
print("Model Tahminleri vs Gerçek Etiketler:")
print(df.head())


plt.figure(figsize=(12, 5)) #geniş bi tuval aç demek istedik

plt.subplot(1, 2, 1) 
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y_true, cmap="viridis")
plt.title("Gerçek Küme Etiketleri")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")

plt.subplot(1, 2, 2) 
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y_pred, cmap="viridis")
plt.title("K-Means'in Bulduğu Kümeler")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")

plt.tight_layout() 
plt.show() 