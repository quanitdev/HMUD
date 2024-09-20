from sklearn.datasets import load_boston
import pandas as pd

# Tải dữ liệu
boston = load_boston()

# Chuyển đổi thành DataFrame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Xem 5 dòng đầu
print(df.head())

# Mô tả thống kê
print(df.describe())