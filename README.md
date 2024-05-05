# EX-09 Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. STOP
2. Import the required packages.
3. Import the dataset to operate on.
4. Split the dataset.
5. Predict the required output.
6. STOP
  
## Program:

```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PPRADEEP V
RegisterNumber: 212223240119
*/
```

```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/spam.csv",encoding='Windows-1252')
data.head()
data.tail()
data.info()
data.isnull().sum()
x=data['v1'].values
y=data['v2'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
# head()
![326372224-a8b9f65e-d8a7-49ec-9ce7-7134dcb8ca28](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/51dca19b-bbda-4fa0-b64c-774b313bcc69)
# tail()
![326372268-2a6c77ef-ee3d-4b92-8271-672e6859333a](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/b520d397-c689-4ebf-88ef-96f35e662aff)
# info()
![326372501-2987c686-bdd6-4046-8b09-3816b6884f4f](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/d124b24d-328b-4590-afec-2202e56d32cf)
# isnull()sum()
![326372607-2f4e8651-f7d3-4f79-9906-f21e9f0592b0](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/45d5cd99-3f1a-48a3-9a33-723a4af68d27)
# y_pred
![326372690-68a82f7f-9cf6-4148-a4fa-b4de3a04339e](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/a1f48dd7-0080-4897-9050-6e8dccb3203c)
# accuracy
![326372725-916da21b-c702-4708-85d1-8fcec998ad04](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/d0d0fa8b-0620-46a7-95b7-b7b12b6e1fd0)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
