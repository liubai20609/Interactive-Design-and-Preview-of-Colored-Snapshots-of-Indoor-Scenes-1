import colormath
import cv2
import csv
import math
import numpy as np
import pandas as pd
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976
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
        RGB = '[' + str(colorRGB[i]) + ',' + str(colorRGB[i + 1]) + ',' + str(colorRGB[i + 2]) + ']'
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

def rgb2lab(rgb):
    r = rgb[0] / 255.0  # rgb range: 0 ~ 1
    g = rgb[1] / 255.0
    b = rgb[2] / 255.0

    # gamma 2.2
    if r > 0.04045:
        r = pow((r + 0.055) / 1.055, 2.4)
    else:
        r = r / 12.92

    if g > 0.04045:
        g = pow((g + 0.055) / 1.055, 2.4)
    else:
        g = g / 12.92

    if b > 0.04045:
        b = pow((b + 0.055) / 1.055, 2.4)
    else:
        b = b / 12.92

    # sRGB
    X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
    Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
    Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470

    # XYZ range: 0~100
    X = X * 100.000
    Y = Y * 100.000
    Z = Z * 100.000

    # Reference White Point

    ref_X = 96.4221
    ref_Y = 100.000
    ref_Z = 82.5211

    X = X / ref_X
    Y = Y / ref_Y
    Z = Z / ref_Z

    # Lab
    if X > 0.008856:
        X = pow(X, 1 / 3.000)
    else:
        X = (7.787 * X) + (16 / 116.000)

    if Y > 0.008856:
        Y = pow(Y, 1 / 3.000)
    else:
        Y = (7.787 * Y) + (16 / 116.000)

    if Z > 0.008856:
        Z = pow(Z, 1 / 3.000)
    else:
        Z = (7.787 * Z) + (16 / 116.000)

    Lab_L = round((116.000 * Y) - 16.000, 2)
    Lab_a = round(500.000 * (X - Y), 2)
    Lab_b = round(200.000 * (Y - Z), 2)

    return [Lab_L, Lab_a, Lab_b]

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


                # HSV_a1, HSV_a2, HSV_a3 = rgb2hsv(a1, a2, a3)      ###########rgb2HSV

                [LAB_a1,LAB_a2,LAB_a3] = rgb2lab([a1,a2,a3])  ##########rgb2LAB

                cha = 1000000000
                # a1, a2, a3 = RGB22HSI(a1, a2, a3)
                for k in range(0, len(colorRGB), 3):
                    # k1,k2,k3= RGB22HSI(colorRGB[k],colorRGB[k+1],colorRGB[k+2])

                    k1,k2,k3= colorRGB[k],colorRGB[k+1],colorRGB[k+2]

                    # LAB颜色空间色差
                    # cha0 = ColourDistance((a1, a2, a3),(k1,k2,k3))
                    # RGB2HSV颜色空间色差
                    # HSV_k1,HSV_k2,HSV_k3=rgb2hsv(k1,k2,k3)###########
                    # cha0 = HSVDistance((HSV_a1, HSV_a2, HSV_a3),(HSV_k1,HSV_k2,HSV_k3))###########

                    #CIEDE2000色差公式
                    [LAB_k1, LAB_k2, LAB_k3] = rgb2lab([k1, k2, k3])  ##########rgb2LAB
                    cha0 = colormath.color_diff.delta_e_cie2000(LabColor(LAB_a1,LAB_a2,LAB_a3), LabColor(LAB_k1, LAB_k2, LAB_k3), Kl=1, Kc=1, Kh=1)


                    # cha0 = ((abs(k1 - a1))**2 + (abs(k2 - a2))**2 + (abs(k3 - a3))**2)**0.5
                    if cha0 <= cha:
                        data[i][j] = int(k/3+1)
                        cha = cha0
            else:
                data[i][j]=0
    return data
#RGB2HSV
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    m = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        if g >= b:
            h = ((g-b)/m)*60
        else:
            h = ((g-b)/m)*60 + 360
    elif mx == g:
        h = ((b-r)/m)*60 + 120
    elif mx == b:
        h = ((r-g)/m)*60 + 240
    if mx == 0:
        s = 0
    else:
        s = m/mx
    v = mx
    return h, s, v
def HSVDistance(hsv_1,hsv_2):
    H_1,S_1,V_1 = hsv_1
    H_2,S_2,V_2 = hsv_2
    R=100
    angle=30
    h = R * math.cos(angle / 180 * math.pi)
    r = R * math.sin(angle / 180 * math.pi)
    x1 = r * V_1 * S_1 * math.cos(H_1 / 180 * math.pi)
    y1 = r * V_1 * S_1 * math.sin(H_1 / 180 * math.pi)
    z1 = h * (1 - V_1)
    x2 = r * V_2 * S_1 * math.cos(H_2 / 180 * math.pi)
    y2 = r * V_2 * S_1 * math.sin(H_2 / 180 * math.pi)
    z2 = h * (1 - V_2)
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return math.sqrt(dx * dx + dy * dy + dz * dz)

#LAB颜色空间色差
def ColourDistance(rgb_1, rgb_2):
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))



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

    墙壁 = label_csv.墙壁.value_counts()
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
file.close()
print("统计的所有家具前十种颜色")
print(jiajuColor)

#转换txt中的rgb为可直接复制粘贴的格式
import re
f = open("jiajuColor.txt",'r+',encoding='utf-8')  # 返回一个文件对象
line = f.read() # 调用文件的 readline()方法
line = re.sub(r'\'','',line)
file = open('jiajuColorChange.txt','w')
file.write(str(line))
f.close()
print("转换txt格式")

#将统计的100个色块的rgb写入txt


def get100RGB3_new():
    final100ColorRGB=[]
    for i in range(0, len(colorRGB), 3):
        RGB = str(colorRGB[i]) + ',' + str(colorRGB[i + 1]) + ',' + str(colorRGB[i + 2])
        final100ColorRGB.append(RGB)
    print('整合的RGB')
    print(final100ColorRGB)
    return final100ColorRGB
final100ColorRGB_new = get100RGB3_new()
file=open("100ColorRGB.txt","w",encoding='utf-8')
for c in range(len(final100ColorRGB_new)-1):
    file.write(str(final100ColorRGB_new[c])+'#')
file.write(final100ColorRGB_new[-1])
print("将统计的100个色块的rgb写入txt")






