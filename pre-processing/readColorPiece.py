import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpig
import numpy as np
picture_path='100color1.jpg'
"""
cv2读取
"""
img=cv2.imread(picture_path)
# cv2.imshow('dog.jpg',img)
# cv2.waitKey(0)
colorRGB = []
for i in range(50,1000,100):
    for j in range(50,1000,100):
        colorRGB.append(img[i, j, 2])
        colorRGB.append(img[i, j, 1])
        colorRGB.append(img[i, j, 0])
print('获取原始图像的R、G、B')
print(colorRGB)
final100ColorRGB=[]
for i in range(0,len(colorRGB),3):
    RGB='('+str(colorRGB[i])+','+str(colorRGB[i+1])+','+str(colorRGB[i+2])+')'
    final100ColorRGB.append(RGB)
print('整合的RGB')
print(final100ColorRGB)
imgCopy = np.zeros([1000, 1000, 3], np.uint8)
k=0
for i in range(0,901,100):
    for j in range(0,901,100):
        imgCopy[i:i+99,j:j+99,2]=colorRGB[k]
        imgCopy[i:i+99,j:j+99,1]=colorRGB[k + 1]
        imgCopy[i:i+99,j:j+99,0]=colorRGB[k + 2]
        k=k+3
# cv2.imshow('dog.jpg',imgCopy)
# cv2.waitKey(0)
cv2.imwrite('gamut.jpg', imgCopy, [int(cv2.IMWRITE_JPEG_QUALITY),100])
print(np.shape(imgCopy))
"""
plt读取
"""
# img=plt.imread(picture_path)
# plt.imshow(img)
# plt.show()
# print(img.shape)
# print(img)
"""
mpig读取
"""
# img=mpig.imread(picture_path)
# plt.imshow(img)
# plt.show()
# print(img.shape)
# print(img)
def rgb2ffffff(value):
    digit = list(map(str, range(10))) + list("abcdef")
    string =''
    a1 = value // 16
    a2 = value % 16
    string += digit[a1] + digit[a2]
    return string
color16=[]
for i in range(0,len(colorRGB)):
    value= rgb2ffffff(colorRGB[i])
    color16.append(value)
print('转为16进制的a、b、c')
print(color16)

final100Color16=[]
for i in range(0,len(color16),3):
    aabbcc=color16[i]+color16[i+1]+color16[i+2]
    final100Color16.append(aabbcc)
print('整合后的16进制aabbcc')
print(final100Color16)