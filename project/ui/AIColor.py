# coding=utf-8
import cv2
import csv
import math
import numpy as np
import pandas as pd
import os

def AIcolor(pos,masksplit256):
    x,y = pos;
    label = masksplit256[y,x,0]
    if label==1:
        print()

#获取100色块的rgb    [0, 0, 0, 25, 25, 25, 50, 50, 50, 76, 76, 76, 102]
def get100RGB():
    picture_path='100color1.jpg'
    img=cv2.imread(picture_path)
    colorRGB = []
    for i in range(50,1000,100):
        for j in range(50,1000,100):
            colorRGB.append(img[i, j, 2])
            colorRGB.append(img[i, j, 1])
            colorRGB.append(img[i, j, 0])
    print('获取原始图像的R、G、B')
    print(colorRGB)
    return colorRGB
#获取100色块的rgb    ['(0,0,0)', '(25,25,25)', '(50,50,50)]
def get100RGB3():
    final100ColorRGB=[]
    for i in range(0, len(colorRGB), 3):
        RGB = '(' + str(colorRGB[i]) + ',' + str(colorRGB[i + 1]) + ',' + str(colorRGB[i + 2]) + ')'
        final100ColorRGB.append(RGB)
    print('整合的RGB')
    print(final100ColorRGB)
    return final100ColorRGB
#rgb转16进制   00-》00  25->19
def rgb2ffffff(value):
    digit = list(map(str, range(10))) + list("abcdef")
    string =''
    a1 = value // 16
    a2 = value % 16
    string += digit[a1] + digit[a2]
    return string
#获取100色块的16进制的r、g、b   ['00', '00', '00', '19', '19', '19', '32', '32', '32']
def getff_ff_ff(colorRGB):
    color16=[]
    for i in range(0, len(colorRGB)):
        value = rgb2ffffff(colorRGB[i])
        color16.append(value)
    print('转为16进制的a、b、c')
    print(color16)
    return color16
#获取100色块的16进制的rrggbb  ['000000', '191919', '323232', '4c4c4c']
def getffffff(colo16):
    final100Color16=[]
    for i in range(0,len(color16),3):
        aabbcc=color16[i]+color16[i+1]+color16[i+2]
        final100Color16.append(aabbcc)
    print('整合后的16进制aabbcc')
    print(final100Color16)
    return final100Color16

#16进制转rgb
def ffffff2rgb(value):
    digit = list(map(str, range(10))) + list("abcdef")
    a1 = digit.index(value[0]) * 16 + digit.index(value[1])
    a2 = digit.index(value[2]) * 16 + digit.index(value[3])
    a3 = digit.index(value[4]) * 16 + digit.index(value[5])
    return a1,a2,a3


colorRGB=get100RGB()                    #获取100色块的rgb    [0, 0, 0, 25, 25, 25, 50, 50, 50, 76, 76, 76, 102]
final100ColorRGB=get100RGB3()           #获取100色块的rgb    ['(0,0,0)', '(25,25,25)', '(50,50,50)]
color16=getff_ff_ff(colorRGB)           #获取100色块的16进制的r、g、b   ['00', '00', '00', '19', '19', '19', '32', '32', '32']
final100Color16=getffffff(color16)      #获取00色块的16进制的rrggbb  ['000000', '191919', '323232', '4c4c4c']


def RGB22HSI(R,G,B):
    H=-1000
    den = math.sqrt((R-G)**2+(R-B)*(G-B))
    if den!=0:
        thetha = np.arccos(0.5 * (R - B + R - G) / den)  # 计算夹角
    else:return R,G,B
    #den>0且G>=B的元素h赋值为thetha
    if den!=0:

        if G >= B:
            H = thetha
        elif G < B:
            H = 360 - thetha
    else:
        H=0
    #计算S通道
    #找出每组RGB值的最小值
    arr = [B,G,R]
    minRGB = min(arr)

    #计算S通道
    S = 1 - minRGB*3/(R+B+G)
    # 计算I通道
    I = (R+G+B)/3
    return H,S,I


#初始化程序
def init():
    with open('搭配统计.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    return rows
#生成新的label.csv
# def create_csv(data):
#     data = data
#     csvFile2 = open('label.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
#     writer = csv.writer(csvFile2)
#     m = len(data)
#     # writer.writerow(data)
#     for i in range(m):
#         writer.writerow(data[i])
#     csvFile2.close()
#
def tolabel(data):
    m = len(data)
    l = len(data[0])
    for i in range(1,m):
        for j in range(1,l):
            if data[i][j]!='':
                a1, a2, a3 = ffffff2rgb(data[i][j])
                cha = 1000000000
                # a1, a2, a3 = RGB22HSI(a1, a2, a3)
                for k in range(0, len(colorRGB), 3):
                    # k1,k2,k3= RGB22HSI(colorRGB[k],colorRGB[k+1],colorRGB[k+2])
                    k1,k2,k3= colorRGB[k],colorRGB[k+1],colorRGB[k+2]
                    cha0 = (abs(k1 - a1))**2 + (abs(k2 - a2))**2 + (abs(k3 - a3))**2
                    if cha0 <= cha:
                        data[i][j] = k/3+1
                        cha = cha0
            else:
                data[i][j]=''
    return data
#RGB2HSI

#ffffff->label
def changelabel(data):
    csvFile2 = open('label.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(csvFile2)
    m = len(data)
    l=len(data[0])
    data=tolabel(data)
    for i in range(m):
         writer.writerow( data[i])
    csvFile2.close()
rows = init()
# create_csv(rows)
changelabel(rows)
# colorValue = input()
# a1,a2,a3=ffffff2rgb(colorValue)
# print(a1,a2,a3)


#获取最后所有家具前10种颜色推荐
def PrintFinalColor():
    label_csv = pd.read_csv('label.csv', encoding='ANSI')
    print('PrintFinalColor方法：label_csv.columns')
    print(label_csv.columns)
    # print(label_csv.index)
    ll = []
    for i in range(1, len(label_csv.columns)):
        # ll.append(label_csv.columns[i])
        a = label_csv[label_csv.columns[i]].value_counts()
        ll.append(a)
    # print(ll)
    # print(ll[1])
    # print(ll[2])

    # print('墙壁墙壁墙壁墙壁墙壁')
    # 墙壁=label_csv.墙壁.value_counts()
    # print(墙壁)
    # print('窗帘窗帘窗帘窗帘窗帘')
    # 窗帘=label_csv.窗帘.value_counts()
    # print(窗帘)
    # print('地板地板地板地板地板')
    # 地板=label_csv.地板.value_counts()
    # print(地板.name)

    # 墙壁 = label_csv.墙壁.value_counts()
    # print(墙壁)

    # 找出每种家具前10种建议颜色
    print('fasd')
    color_level = label_csv['墙壁'].value_counts()._stat_axis[0]
    color_level_sum = label_csv['墙壁'].value_counts().array[1]
    # print(color_level)
    # print(color_level_sum)
    jiajuColor = {}
    for i in range(1, len(label_csv.columns)):
        color_mid=[]
        a = label_csv[label_csv.columns[i]].value_counts()
        # jiajuColor.append(label_csv.columns[i])
        for j in range(len(a._stat_axis)):
            rgbPiece = final100ColorRGB[int(a._stat_axis[j])-1]
            color_mid.append(rgbPiece)
        jiajuColor[label_csv.columns[i]]=color_mid
    return jiajuColor

jiajuColor=PrintFinalColor()

#将统计的家具前十种颜色写入txt
file=open("jiajuColor.txt","w",encoding='utf-8')
label_csv = pd.read_csv('label.csv', encoding='ANSI')
for i in range(1, len(label_csv.columns)):
    file.write(str(label_csv.columns[i])+'\n'+str(jiajuColor[label_csv.columns[i]])+'\n')
print("统计的所有家具前十种颜色")
print(jiajuColor)








