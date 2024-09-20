import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
dataset=pd.read_csv("winequality-red.csv")
dataset.head()
dataset.shape
dataset.isnull().sum()
plt.figure(figsize=(8,8))
sns.heatmap(dataset.corr(), cbar=True, square=True, fmt=".1f", annot=True, annot_kws={'size':8}, cmap='coolwarm')
x=dataset.drop(['quality'], axis=1).values
print(x)
y=dataset['quality'].apply(lambda y_value:1 if y_value>=7 else 0).values
print(y)
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=2)
x.shape, x_train.shape, x_test.shape
model=RandomForestClassifier()
model.fit(x_train, y_train)
predict=model.predict(x_train)
accuracy=accuracy_score(predict, y_train)
print("Accuracy of train Dataset --> ", np.multiply(accuracy, 100), "%")
predict=model.predict(x_test)
accuracy=accuracy_score(predict, y_test)
print("Accuracy of test Dataset --> ", np.multiply(accuracy, 100), "%")
input=(7.4,	0.70,	0.00,	1.9,	0.076,	11.0,	34.0,	0.9978,	3.51,	0.56,	9.4	)
input_data_as_numpy_array=np.asarray(input)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
if prediction[0]==1:
  print("Good Quality Wine")
else:
  print("Bad Quality Wine")