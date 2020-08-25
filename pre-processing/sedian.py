import pandas as pd
import cv2
import numpy as np
import csv


# img = np.zeros([74, 74, 3], np.uint8)
# img[:, :, 0] = 255
# img[:,:, 1] = 0
# img[:,:, 2] = 0
# cv2.imwrite('sedian.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY),100])
# print('ok')




def init():
    with open('搭配统计.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    return rows
def init_label():
    with open('label.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        rows_label = [row for row in reader]
    return rows_label
def txtReady():
    file = open('100ColorRGB.txt', 'r')
    file = file.read().split('#')
    new100color = []
    for i in range(len(file)):
        new100color.append(file[i].split(','))
    return new100color

def ffffff2rgb(value):
    digit = list(map(str, range(10))) + list("abcdef")
    a1 = digit.index(value[0]) * 16 + digit.index(value[1])
    a2 = digit.index(value[2]) * 16 + digit.index(value[3])
    a3 = digit.index(value[4]) * 16 + digit.index(value[5])
    return [a1,a2,a3]
new100color = txtReady()
rows = init()
rows_label = init_label()

imgCopy = np.zeros([7400, 7400, 3], np.uint8)
imgCopy_label = np.zeros([7400, 7400, 3], np.uint8)
i=0
j=0
for m in range(1,len(rows)):
    for n in range(1,len(rows[0])):
                if rows[m][n]!='':
                    colorRGB = ffffff2rgb(rows[m][n])
                    imgCopy[i:i + 99, j:j + 99, 2] = colorRGB[0]
                    imgCopy[i:i + 99, j:j + 99, 1] = colorRGB[1]
                    imgCopy[i:i + 99, j:j + 99, 0] = colorRGB[2]

                    colorRGB_label = new100color[int(rows_label[m][n]) - 1]
                    imgCopy_label[i:i + 99, j:j + 99, 2] = int(colorRGB_label[0])
                    imgCopy_label[i:i + 99, j:j + 99, 1] = int(colorRGB_label[1])
                    imgCopy_label[i:i + 99, j:j + 99, 0] = int(colorRGB_label[2])

                    j = j+100
                    if j>7350:
                        i = i + 100
                        j=0
                else:
                    imgCopy[i:i + 99, j:j + 99, 2] = 0
                    imgCopy[i:i + 99, j:j + 99, 1] = 0
                    imgCopy[i:i + 99, j:j + 99, 0] = 0

                    imgCopy_label[i:i + 99, j:j + 99, 2] = 0
                    imgCopy_label[i:i + 99, j:j + 99, 1] = 0
                    imgCopy_label[i:i + 99, j:j + 99, 0] = 0

                    j = j + 100
                    if j > 7350:
                        i = i + 100
                        j = 0




# cv2.imshow('sedian.jpg',imgCopy)
# cv2.waitKey(0)
cv2.imwrite('sedian.jpg', imgCopy, [int(cv2.IMWRITE_JPEG_QUALITY),100])
cv2.imwrite('sedian_label_LAB_new100-2000.jpg', imgCopy_label, [int(cv2.IMWRITE_JPEG_QUALITY),100])
print('ok')