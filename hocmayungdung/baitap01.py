import numpy as np
v1 = np.arange(15, dtype=int)
v2 = np.arange(start=-5, stop =
10, dtype=int)
print(v1)
print(v2)

#Tạo 1 vector các giá trị số thực
# tăng dần và
#cách đều nhau
v3 = np.linspace(start=-2, stop=
3, num=15, dtype=float)
print(v3)

#Sinh ngẫu nhiên 1 vector số
# nguyên 10 phần tử
#giá trị mỗi phần tử là 0 hoặc 1
np.random.seed(123)
v4 = np.random.randint(2, size=10)
print(v4)
#Sinh ngẫu nhiên 1 vector số
# nguyên 10 phần tử
#giá trị mỗi phần tử ngẫu nhiên
# trong -3, 3
v5 = np.random.randint(low=-3,
high=3, size=10)
print(v5)
