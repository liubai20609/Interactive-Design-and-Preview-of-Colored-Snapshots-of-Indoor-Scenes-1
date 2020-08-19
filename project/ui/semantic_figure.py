# coding=utf-8
# 黄色检测
import numpy as np
import cv2
import os

def semantic_figure(path):
    aaa = np.zeros((256, 256, 3), np.uint8)
    semantic_path=path+"/yuyitu/yuyitu.png"
    image = cv2.imread(semantic_path)
    # image = cv2.resize(image, (512, 288), interpolation=cv2.INTER_CUBIC)
    color = [
        ([0, 250, 0], [0, 250, 0]), ([125, 125, 62], [125, 125, 62]),
        ([61, 124, 250], [61, 124, 250]),([250, 0, 0], [250, 0, 0]),
        ([240, 29, 250],[240, 29, 250]),([147,98,10],[147,98,10]),
        ([0,250,222],[0,250,222]),([250,0,134],[250,0,134]),
        ([0,147,248],[0,147,248]),([117,250,0],[117,250,0]),
        ([0,189,241],[0,189,241]),([128,128,255],[128,128,255]),
        ([64,64,0],[64,64,0]),([255,128,128],[255,128,128]),
        ([64,128,0],[64,128,0]),([64,0,64],[64,0,64]),
        ([192,208,49],[192,208,49]),([200,50,200],[200,50,200])
        #地板、椅子、桌台、地毯 注意：数值按[b,g,r]排布
    ]
    h,w,c=image.shape
    mask_split=np.zeros((h, w, 3), np.uint8)
    # 如果color中定义了几种颜色区间，都可以分割出来
    i=1
    for (lower, upper) in color:
        # 创建NumPy数组
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限
        # 根据阈值找到对应颜色
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        #转黑白
        logo_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(logo_gray, 1, 255, cv2.THRESH_BINARY)
        rows, cols, channels = output.shape
        # 腐蚀膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # ¾ØÐÎ½á¹¹
        erode = cv2.dilate(mask, kernel, iterations=1)
        # erode = cv2.erode(dilate, None, iterations=1)
        mask_split[:,:,0]=(~(erode > 0))*mask_split[:,:,0]+(erode > 0) * i
        i=i+1

    mask_split[:, :, 1] = mask_split[:, :, 0]
    mask_split[:, :, 2] = mask_split[:, :, 0]
    # # 平滑
    # kernel = np.ones((7, 7), np.float32) / 25
    # mask_split = cv2.filter2D(mask_split, -1, kernel)
    if not os.path.exists(path + "/yuyitu/mask_split.png"):
        cv2.imwrite(path + '/yuyitu/mask_split.png', mask_split)

    mask_split256 = cv2.resize(mask_split, (256, 256), interpolation=cv2.INTER_NEAREST)
    print("semantic_figure:ok")
    if os.path.exists(path + "/yuyitu/Allmask.png"):
        virtualMask = cv2.imread(path + "/yuyitu/Allmask.png")
    else:
        virtualMask = virtualPiont(path,mask_split256)   #生成虚拟mask

    # cv2.imshow('allMask',virtualMask)
    # cv2.waitKey(0)
    #装饰家具

    #获取白色、黑色家具
    white,black = white_black(path)

    zhuangshi = cv2.imread(path + "/yuyitu/zhuangshi.png")
    # zhuangshi = cv2.resize(zhuangshi, (512, 288), interpolation=cv2.INTER_CUBIC)
    # zhuangshi = cv2.cvtColor(zhuangshi, cv2.COLOR_BGR2RGB)
    return mask_split,i-1,mask_split256,virtualMask,zhuangshi,white,black


def virtualPiont(path,mask_split256):
    zeros = np.zeros([256, 256, 3], np.uint8)
    for label in range(1, 18):
        image = mask_split256
        one = np.ones([256, 256, 3], np.uint8)
        image = (image[:, :, 0] == label) * one[:, :, 0] * 255
        img = image.copy()
        # cv2.imshow('23', image)
        # cv2.waitKey(0)
        sumValue = np.sum(image) / 255
        if int(sumValue) == 0:
            continue
        k = 5
        if sumValue > 16000:
            N = 12
            mn = 19
        elif sumValue > 10000 and sumValue <= 16000:
            N = 10
            mn = 19
        elif sumValue <= 10000 and sumValue > 5000:
            N = 8
            mn = 15
        elif sumValue <= 5000 and sumValue > 2000:
            N = 6
            mn = 9
            k = 4
        elif sumValue <= 2000 and sumValue > 1000:
            N = 4
            mn = 9
            k = 3
        else:
            N = 4
            mn = 3
            k = 2
        print(sumValue)

        # 超像素分割
        # segments = slic(image, n_segments=200, compactness=10)
        # out=mark_boundaries(image,segments)
        # plt.subplot(121)
        # plt.title("n_segments=60")
        # plt.imshow(out)
        #
        # segments2 = slic(image, n_segments=300, compactness=10)
        # out2=mark_boundaries(image,segments2)
        # plt.subplot(122)
        # plt.title("n_segments=300")
        # plt.imshow(out2)
        # plt.show()

        xy = []
        for i in range(int(N)):
            dict = {}
            for m in range(mn, 256 - mn):
                for n in range(mn, 256 - mn):
                    sumV = np.sum(image[m - mn:m + mn + 1, n - mn:n + mn + 1])
                    dict[(m, n)] = sumV
            res = sorted(dict.items(), key=lambda dict: dict[1], reverse=True)
            # print(res[0][0])
            newM = res[0][0][0]
            newN = res[0][0][1]
            while int(np.sum(image[newM:newM + k, newN:newN + k])) != 255 * (k) * (k):
                if newM < 255:
                    newM = newM + 1
                else:
                    if newN < 255:
                        newN = newN + 1
                    else:
                        break

            xy.append((newM, newN))
            image[newM - mn:newM + mn + 1, newN - mn:newN + mn + 1] = 0

        for i in range(len(xy)):
            center_x = xy[i][0]
            center_y = xy[i][1]
            # cv2.circle(img, (center_x, center_y), 7, 128, -1)#»æÖÆÖÐÐÄµã
            img[center_x:center_x + k, center_y:center_y + k] = 128
            one[center_x:center_x + k, center_y:center_y + k, :] = 255
            zeros[center_x:center_x + k, center_y:center_y + k, :] = 255

        cv2.imwrite(str(label) + '.png', img)
        cv2.imwrite("text.png", one)
        cv2.imwrite("Allmask.png", zeros)

        print(N)
    cv2.imwrite(path+'/yuyitu/Allmask.png',zeros)
    return zeros

def white_black(path):
    white =  cv2.imread(path + "/yuyitu/white.png")
    black =  cv2.imread(path + "/yuyitu/black.png")
    return white,black
