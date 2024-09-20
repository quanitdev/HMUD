import  numpy as np
from  sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

#Buoc 1: Tai du lieu
D = load_iris()
X = D.data
y = D.target
#Buoc 2: Phan chia du lieu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=105)
print("Kich thuoc tap du lieu goc")
print("X: ", X.shape, "; y: ", y.shape)
print("Kich thuoc tap du lieu huan luyen - training dataset")
print("X_train: ", X_train.shape, "; y_train: ", y_train.shape)
print("Kich thuoc tap du lieu kiem thu - test dataset")
print("X_test: ", X_test.shape, "; y_test: ", y_test.shape)
#Buoc 3: Huan luyen mo hinh Naive Bayes voi tap du lieu Iris
model = LinearSVC()
model.fit(X_train, y_train)
#Buoc 4: Su dung mo hinh de du doan
y_hat = model.predict(X_test)
#Buoc 5: Danh gia hieu nang du doan cua mo hinh
print("Accuracy score: ", accuracy_score(y_test, y_hat))
print("Classification report: ")
print(classification_report(y_test, y_hat))
cf_matrix = confusion_matrix(y_test, y_hat)
print("Confusion matrix: ")
print(cf_matrix)

#Buoc 6: Luu lai mo hinh
with open('E:\HMUD\hocmay\mohinhsvm\svmiris.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Da luu mo hinh thanh cong...")

