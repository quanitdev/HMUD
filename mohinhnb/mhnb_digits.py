import numpy as np
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

#buoc 1 tai du lieu
D= load_digits()
X= D.data
y=D.target

#buoc 2 pha chia du lieu
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=105)
print("Kich thuoc tap du lieu goc")
print("X: ",X.shape, "; y: ",y.shape)
print("Kich thuoc tap du lieu huan luyen - traning dataset")
print("X_train: ",X_train.shape, "; y_train: ",y_train.shape)
print("Kich thuoc tap du lieu kiem thu - test dataset")
print("X_test: ",X_test.shape, "; y_test: ",y_test.shape)

#Buoc 3 Huan luyen mo hinh Naive Bayes voi tap du lieu Iris
model = GaussianNB()
model.fit(X_train, y_train)

#Buoc 4: Su dung mo hinh de du doan

y_hat= model.predict(X_test)
#Buoc 5: Danh gia hieu nang du doan cua mo hinh
print("Accuracy score: ", accuracy_score(y_test, y_hat))

#Buoc 6 luu mo hinh
with open('E:/HMUD/hocmay/mohinhnb/mhnb_digits.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Da luu mo hinh thanh cong...")