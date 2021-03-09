import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

csv = 'weatherAUS.csv'

def GetDataset(csv):
    dt = pd.read_csv(csv)
    return dt

dataset =GetDataset(csv)

dependent = dataset.iloc[:10, -1]

independent = dataset.iloc[:10, [2, 3, 7, 8, 11]].dropna()

transformer = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[2])], remainder='passthrough')

independent = np.array(transformer.fit_transform(independent))
#print(independent)

ind_train, ind_test, dep_train, dep_test = train_test_split(independent,dependent, test_size = 0.15, random_state=0)

linearRegression  = LinearRegression()
linearRegression.fit(ind_train,dep_train)

dep_pred = linearRegression.predict(ind_test)

np.set_printoptions(precision=2)

dep_pred_col = dep_pred.reshape(len(dep_pred),1)
dep_test_col = dep_pred.reshape(len(dep_test),1)

print(np.concatenate((dep_pred_col,dep_test_col), axis=1 ))

for c in linearRegression.coef_:
    print (f'{c:.2f} ')
print (f'{linearRegression.intercept_:.2f}')


