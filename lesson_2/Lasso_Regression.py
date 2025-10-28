import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso 

df = sns.load_dataset("tips")
y = df['tip']
X = df[["total_bill", "size", "sex", "smoker", "day", "time"]]
num_cols = ["total_bill", "size"]
cat_cols = ["sex", "smoker", "day", "time"]

preprocessor = ColumnTransformer(#regression_pipline.py sınııf ile aynı mantık
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown='ignore'), cat_cols),
        ("num", "passthrough", num_cols)
    ],
    remainder='drop'
)

model_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                 ("model", Lasso(alpha=0.1))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

print(f"Lasso Model R2 Skoru: {model_pipeline.score(X_test, y_test):.4f}")# %52

model_coefficients = model_pipeline.named_steps['model'].coef_
print(f"\nModel Katsayıları (Coefficients):\n{model_coefficients}")#önemsiz özellikleri modelden tamamen atar -katsayılarını 0 yapar-.