import  numpy as np
from  sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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
model = MLPClassifier(hidden_layer_sizes=(50, 15, 50), max_iter=10)
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
with open('E:/HMUD/hocmay/mohinhann/anniris.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Da luu mo hinh thanh cong...")
#Buoc 7: Ve do thi nhiet
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận hỗn hợp')
plt.show()
#Buoc 8: Ve Loss curve
# Lấy giá trị loss (nếu có)
loss_values = model.loss_curve_  # Giả sử có thuộc tính này

# Vẽ đồ thị
plt.plot(loss_values)
plt.title('Đồ thị Loss')
plt.ylabel('Loss')
plt.xlabel('Số lần lặp')
plt.show()
#Buoc 9: Ve duong ROC
from sklearn.metrics import roc_curve, auc
# Dự đoán trên tập kiểm tra
y_score = model.predict_proba(X_test)
# Số lượng lớp
n_classes = len(np.unique(y))

# Vẽ đường ROC cho từng lớp
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, thresholds = roc_curve(y_test == i, y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()