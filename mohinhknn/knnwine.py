from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Buoc1: tai du lieu
D=load_wine()
X=D.data
y=D.target
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=15)
print("kich thuoc tap du lieu kiem thu")
print("X_test= ",X_test.shape)
print("y_train= ",y_test.shape)
print("kich thuoc tap du lieu huan luyen")
print("X_train= ",X_train.shape)
print("y_train= ",y_train.shape)
print("kich thuoc tap du lieu kiem thu")
print("X_test= ",X_test.shape)
print("y_train= ",y_test.shape)


#Buoc 2 Huan luyen mo hinh voi k=3

model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)

#Buoc 3: su dung mo hinh de du doan
y_hat = model.predict(X_test)

#Buoc 4 Danh gia hieu naang du doan cua mo hinh
print("Accuracy score= ", accuracy_score(y_test,y_hat))


#Buoc 5 luu mo hinh
import joblib
joblib.dump(model, 'E:\HMUD\hocmay\mohinhknn\modelk5wine.joblib')
print("Luu mo hinh thanh cong...")