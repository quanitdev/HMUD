import numpy as np
from skimage import io
from matplotlib import pyplot as plt
imgPath = "F:\HMUD\hocmay\hue.jpg"
img = io.imread(imgPath)
# in kich thuoc anh mau
print(img.shape)

print(("Ma tran anh mau"))
print(img)
print("Lay ma tran anh red")
imgRed= img[:,:,0]
print("gia tri ma tran anh red")
print(imgRed)
io.inshow(imgRed)
plt.show()

print("Lay ma tran anh green")
imgGreen= img[:,:,1]
io.inshow(imgGreen)
plt.show()
#Tao anh xam
img = io.imread(imgPath, as_gray=True)
print(img.shape)
#in anh
io.imshow(img)
plt.show()

# in gia tri cua ma trong anh
print(img)
# in vecto thu 7 trong ma tran anh
print("In vecto thu 7 trong anh: ")
print(img[6])
