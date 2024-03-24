import pandas as pd
from joblib import load

test_samples = pd.read_csv("test_samples.csv")

X_test = test_samples.drop("cardio", axis=1)

bästa_modell = load("bästa_modell.pkl")

förutsägelser = bästa_modell.predict(X_test)

sannolikheter = bästa_modell.predict_proba(X_test)

resultat = pd.DataFrame({
    "probability class 0": sannolikheter[:, 0],
    "probability class 1": sannolikheter[:, 1],
    "predictions": förutsägelser
})

resultat.to_csv("prediction.csv", index=False)