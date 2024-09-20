import pandas as pd
import numpy as np

#Buoc 1: Doc du lieu Boston house price dataset vao bo nho
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
print("Kich thuoc tap thuoc tinh X = ", X.shape)
print("Kich thuoc tap gia tri dich y = ", y.shape)
#Buoc 2: Huan luyen mo hinh HQTT voi tap du lieu tren
from sklearn.linear_model import LinearRegression
#Khởi tạo mô hình
model = LinearRegression()
#huấn luyện mô hình
model.fit(X, y)
#Buoc 3: Lưu mô hình vào file
import pickle
# Lưu mô hình vào file
with open('E:\HMUD\hocmay\HQTT\model02.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Luu mo hinh thanh cong...")