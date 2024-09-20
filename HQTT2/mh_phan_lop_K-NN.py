from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#lấy dữ liệu train_test
iris= load_iris()
X_train, X_test, y_train,  y_test= train_test_split(iris.data,iris.target,test_size=0.3, random_state=42)
#xây dựng mô hình k_NN với k= 7
knn= KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
#đánh giá mô hình
y_pred= knn.predict(X_test)
print((metrics.accuracy_score(y_test, y_pred)))
#Lưu mô hinh
import joblib
joblib.dump(knn,"knn-model.pkl")
print("model has knn-model.pkl")