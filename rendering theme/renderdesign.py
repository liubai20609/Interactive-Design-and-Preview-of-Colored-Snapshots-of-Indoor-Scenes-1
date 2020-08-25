import csv
jiaju = ['桌子', '椅子', '地板', '地毯', '墙壁', '橱柜', '床', '电视柜', '书柜', '窗帘', '床被', '靠枕', '电视墙', '天花板', '沙发', '门', '茶几']
print('Please refer to "connection relation.png" for furniture category')
print('Please enter the furniture in the scene'+str(jiaju))
furniture = []
T = True
def input_furniture():
    global furniture
    furniture = list(map(str, input().split()))
    for i in furniture:
        if i not in jiaju:
            print('Furniture input error, please re-enter：')
            input_furniture()
            break
    return furniture
# furniture = input_furniture()
input_furniture()
print(furniture)
furniture_numpy = []
with open('renderingtheme.csv','r') as f:
    reader = csv.reader(f)
    for row in reader:
        furniture_numpy.append(row)
render_num = 0
k=0
furniture_num = len(furniture_numpy[0])
furniture_index = []
for i in range(len(furniture)):
    for j in range(furniture_num):
        if str(furniture[i])==str(furniture_numpy[0][j]):
            num = int(furniture_numpy[1][j])
            furniture_index.append(j)
            if k<num:
                k = num
            continue
print('The number of rendered grayscale images is ：'+str(k))
renderdesign = []
for i in range(k):
    theme = 'Rendering theme'+ str(i+1) + "=="
    for j in range(len(furniture_index)):
        theme += furniture_numpy[0][furniture_index[j]]+"-"+furniture_numpy[i+2][furniture_index[j]]+"  "
    renderdesign.append(theme)


def write_txt(title_name):      #写入txt
    with open('Rendering theme.txt', 'w') as file_handle:  # .txt可以不自己新建,代码会自动新建
        for i in range(len(title_name)):
            file_handle.write("{}\n".format(title_name[i]))  # 此时不需在第2行中的转为字符串
        file_handle.close()


write_txt(renderdesign)
for i in renderdesign:
    print(i)