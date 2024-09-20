import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Buoc 1: Doc du lieu Boston house price dataset vao bo nho
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
print("Kich thuoc tap thuoc tinh X = ", X.shape)
print("Kich thuoc tap gia tri dich y = ", y.shape)
#Buoc 2: Phan chia tap du lieu train - test theo ti le 80 - 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=15)
#Buoc 3: Huan luyen mo hinh HQTT voi tap du lieu train
from sklearn.linear_model import LinearRegression
#Khởi tạo mô hình
model = LinearRegression()
#huấn luyện mô hình
model.fit(X_train, y_train)
#Buoc 4: Su dung mo hinh de du doan
y_hat = model.predict(X_test)
#Buoc 5: danh gia hieu nang du doan cua mo hinh
from sklearn.metrics import mean_squared_error
print("MSE(y_test, y_hat) = ", mean_squared_error(y_test, y_hat))
#Buoc 6: Lưu mô hình vào file
import joblib
# Lưu mô hình vào file
joblib.dump(model, 'E:\HMUD\hocmay\HQTT2\model_lg.joblib')
print("Luu mo hinh thanh cong...")