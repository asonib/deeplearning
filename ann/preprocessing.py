import pandas as pd

data = pd.read_csv('./Churn_Modelling.csv')
data.head(5)

X = data.iloc[:, 3:13].values
y = data.iloc[:, -1].values

pd.DataFrame(X).head(5)
pd.DataFrame(y).head(3)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X[:, 1] = encoder.fit_transform(X[:, 1])
X[:, 2] = encoder.fit_transform(X[:, 2])

pd.DataFrame(X).head(5)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
pd.DataFrame(X).head(5)

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pd.DataFrame(X_train).head(5)
pd.DataFrame(X_test).head(5)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_train.to_csv('data/training/training_data_features.csv')
y_train.to_csv('data/training/training_data_target.csv')

X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)
X_test.to_csv('data/validation/validation_data_features.csv')
y_test.to_csv('data/validation/validation_data_target.csv')


