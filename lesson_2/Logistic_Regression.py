import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = sns.load_dataset("tips")

X = df[["total_bill"]]

y_bin = (df["tip"] > df["tip"].median()).astype(int)#her bir bahşiş değerini bahşişlerin orta değeriyle karşılaştırır ve büyükse 1 küçükse 0 değerini atar

X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"Lojistik Regresyon Skoru (Accuracy): {model.score(X_test, y_test):.4f}") #%75 doğruluk