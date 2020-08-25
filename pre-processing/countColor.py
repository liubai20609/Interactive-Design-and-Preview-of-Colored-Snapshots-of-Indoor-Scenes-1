import csv
import numpy as np
import pandas as pd

#计算条件概率
label_csv = pd.read_csv('label.csv', encoding='ANSI')
k = len(label_csv._info_axis)-1    #家具类别数量17
countColor = np.zeros([k+1,k+1,101,101],np.uint8)
n = label_csv.shape[0]              #获取的场景样本数322
for t in range(n):
    scene = label_csv.__array__()[t][1:]
    for i in range(k):
        for j in range(k):
            if scene[i] != 0 and scene[j] != 0:
                C1 = int(scene[i])
                C2 = int(scene[j])
                countColor[i+1,j+1,C1,C2] +=1
colorCountNum = countColor
np.save('colorCountNum.npy',colorCountNum)
countColor = countColor/n
np.save('colorCount.npy',countColor)        #countColor存储 [家具label-1,颜色label-1]
print(k)