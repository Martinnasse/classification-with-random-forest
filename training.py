# %%
import pandas as pd

df = pd.read_csv("titanic.csv", sep="\t")

# %%
null = df.apply(lambda x: x.isnull().sum() / len(x) * 100).reset_index()

# %%
# Data processing
import numpy as np


def data_processing(df):
    df.loc[df.Age.isnull(), "Age"] = df.Age.mean()
    df.loc[df.Embarked.isnull(), "Embarked"] = df.Embarked.value_counts().index[0]
    df["Cabin_Is_filled"] = df.Cabin.apply(lambda x: 0 if str(x) == "nan" else 1)
    df = df.drop("Cabin", axis=1)
    df = df.drop("Name", axis=1)
    df = df.drop("Ticket", axis=1)
    df = df.drop("PassengerId", axis=1)
    return df


df=data_processing(df)


# %%
# Transform categorical variables
def features_engineering(df):
    categorical_variables = ["Pclass", "SibSp", "Parch", "Sex", "Embarked"]
    for i in categorical_variables[:3]: df[i] = df[i].apply(lambda x: str(int(x)))
    categorical_data = pd.get_dummies(df[categorical_variables])
    df = pd.concat([df[["Survived", "Age", "Fare"]], categorical_data], axis=1)
    return df


df=features_engineering(df)

# %%
# Create train test dataset
from sklearn.model_selection import train_test_split

y_train, y_test, x_train, x_test = train_test_split(df.Survived, df[df.columns[1:]], test_size=0.33)

# %%
# training
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# %%
# prediction
y_pred = model.predict(x_test)

# %%
# Performance
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
