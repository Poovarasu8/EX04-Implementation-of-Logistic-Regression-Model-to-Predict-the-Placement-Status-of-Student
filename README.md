# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.x is the feature matrix, and y is the target variable

2.train_test_split splits the data.

3.LogisticRegression builds the model.

4.accuracy_score evaluates performance.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Poovarasu V 
RegisterNumber:2305002017  
*/
import pandas as pd
d=pd.read_csv("/content/ex45Placement_Data.csv")
d.head()
d1=d.copy()
d1.head()
d1=d1.drop(['sl_no','salary'],axis=1)
d1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
d1["gender"]=le.fit_transform(d1["gender"])
d1["ssc_b"]=le.fit_transform(d1["ssc_b"])
d1["hsc_b"]=le.fit_transform(d1["hsc_b"])
d1["hsc_s"]=le.fit_transform(d1["hsc_s"])
d1["degree_t"]=le.fit_transform(d1["degree_t"])
d1["workex"]=le.fit_transform(d1["workex"])
d1["specialisation"]=le.fit_transform(d1["specialisation"])
d1["status"]=le.fit_transform(d1["status"])
d1
x=d1.iloc[:,:-1]
x
y=d1.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
a=accuracy_score(y_test,y_pred)
c=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("accuracy score:",a)
print("\nconfusion matrix:\n",c)
print("\nclssification report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=c,display_labels=[True,False])
cm_display.plot()
```

## Output:
![Screenshot (39)](https://github.com/user-attachments/assets/bf8ee0e7-8a89-4b98-bd36-b3dfdcebddb2)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
