import numpy as np
import os

import colormath
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976

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

def txtReady():
    file = open('./myfile/100ColorRGB.txt', 'r')
    file = file.read().split('#')
    new100color = []
    for i in range(len(file)):
        new100color.append(file[i].split(','))
    return new100color
def findclosestColor_label(a):
    new100color = txtReady()
    cha = 1000000000
    lable=0
    for i in range(100):
        r,g,b = int(new100color[i][0]),int(new100color[i][1]),int(new100color[i][2])
        R,G,B = a[0],a[1],a[2]

        [LAB_r, LAB_g, LAB_b] = rgb2lab([r, g, b])
        [LAB_R, LAB_G, LAB_B] = rgb2lab([R, G, B])

        cha0 = colormath.color_diff.delta_e_cie2000(LabColor(LAB_r, LAB_g, LAB_b), LabColor(LAB_R, LAB_G, LAB_B),
                                                    Kl=1, Kc=1, Kh=1)
        # cha0 = (r-R)**2+(g-G)**2+(b-B)**2
        if cha0<=cha:
            lable=i
            cha = cha0
    return lable+1


def findNeedColor(rgb):
    colorPiece = txtReady()
    countColor = np.load('./myfile/colorCount.npy')
    l = {}
    for i in range(1, 18):
        l[i] = 0
    l.update(rgb)
    color_p = {}
    needColorLabel = {}
    for k in range(1, 18):
        if l[k] == 0:
            for i in range(1, 101):#100color-bin
                p = 0
                for x in range(1, len(l)):
                    if l[x] != 0:
                        p += countColor[k, x, i, l[x]]
                color_p[i] = p
            sort_color_p = sorted(color_p.items(), key=lambda d: d[1], reverse=True)
            needColorLabel[k] = colorPiece[sort_color_p[0][0]-1]
    return needColorLabel