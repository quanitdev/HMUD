#import các thư viện cần thiết
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
#tải dữ liệu Boston housing price
boston = load_boston()
#chia dữ liệu thành tập huấn luyện  và tập kiểm thử
X_train, X_test, y_train, y_test= train_test_split(
    boston.data, boston.target, test_size=0.25, random_state=42)
# in ra kích thước của tập huến luyện và tập kiểm thử
print("kích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm thử:", X_test.shape)