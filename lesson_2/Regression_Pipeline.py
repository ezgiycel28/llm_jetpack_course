import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

df = sns.load_dataset("tips")

y = df['tip']
X = df[["total_bill", "size", "sex", "smoker", "day", "time"]]

num_cols = ["total_bill", "size"] #sayısal
cat_cols = ["sex", "smoker", "day", "time"] # kategorik

preprocessor = ColumnTransformer(#elimizdeki veri setini alır ve hangi sütuna hangi işlemin yapılacağını organize eder
    transformers=[
        #0-1 arası değerler verir
        ("cat", OneHotEncoder(drop="first", handle_unknown='ignore'), cat_cols), 
        ("num", "passthrough", num_cols)#geç dokunma demektir
    ],
    remainder='drop'#geri kalanı at sil demektir
)

# veriyi önce 'preprocessor'dan geçirir, sonra 'model'e sokar.
model_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                 ("model", LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

# R2 skoru 1'e ne kadar yakınsa o kadar iyidir.
print(f"Modelin R2 Skoru (Başarısı): {model_pipeline.score(X_test, y_test):.4f}")