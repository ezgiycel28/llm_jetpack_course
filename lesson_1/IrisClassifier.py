                                 #BASİT MAKİNE ÖĞRENMESİNE GİRİŞ#

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets

iris = datasets.load_iris() #sklearn kütüphanesi içindeki hazır veri seti
x = iris.data #girdi
y = iris.target #çıktı

df = pd.DataFrame(x, columns=iris.feature_names)
df['target'] = y
print(df.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler() #veriyi standartlaştıran bir araç
x_train_scaled = scaler.fit_transform(x_train) #ortalama ve standard sapma değerlerini öğrenir ve x_train verisine dönüştürür

x_test_scaled = scaler.transform(x_test) #öğrenmemesi gerektiği için fit_transform kullanmıyoruz.

model = LogisticRegression(max_iter=200)
model.fit(x_train_scaled, y_train)#ölçeklendirilmiş girdiye bakar ve çıktısının ne olduğunun öğrenir

y_pred = model.predict(x_test_scaled)#modelin tahminlerini içeren lste

acc = accuracy_score(y_test, y_pred)#gerçek değerle tahmin edilen değerleri karşılaştır ve bi oran ver demektir
print(f"\nDoğruluk (accuracy): {acc:.4f} \n")

print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))