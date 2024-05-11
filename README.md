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
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')
from sklearn.model_selection import train_test_split
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 
accuracy=accuracy_score(y_test,y_pred)
accuracy
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)

```
## Output:
# data
![329567581-dff3e622-e7b8-4a15-9c18-e39caf1d06fa](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/def078f4-f1ef-41d2-9d2d-5d3466be9ed9)
# data.shape:
![329567625-46019e0a-abd4-45ac-9793-ad85d5fc4f81](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/fc6f4d96-05fb-42c5-ab1c-0ee16abfae9f)
# Y.shape:
![329567862-cdbf779c-5abe-426a-9a43-06fabd253930](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/02b607fc-7151-42e8-83dd-0b8d127b8be9)
# x_train
![329567724-c5d76edd-4d65-4b34-94c8-bd14e53e21e3](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/e148affd-1d7f-4f21-9c7c-4e11f5ebeac8)
# x_train.shape:
![329567953-0a132e17-a840-4ae2-955e-b866efcfd4ae](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/44a22446-4228-4f86-b699-46d8ecb2fcc1)
# y_pred:
![329568013-f3edd0ee-06fb-4594-b99e-06a761fe9956](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/2a42b825-0c15-4837-ae04-d77a81c79351)
# Accuracy:
![329568044-55accff0-f33c-4575-9b70-ab39e7ca51d4](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/7bb9ed7f-5e06-491f-9d09-0c2a21506cb6)
# confusion_matrix:
![329567542-0196f78d-660e-4592-9b61-165c43dd281d](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/27c5205b-9e9d-4b66-a56d-277551f5b586)
# classification_report:
![329568557-02df2736-a171-4c0d-8a81-4fd5ce0f2c73](https://github.com/velupradeep/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150329341/8333f0e1-4f80-4ab7-a5af-a0c0469786bd)







## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
