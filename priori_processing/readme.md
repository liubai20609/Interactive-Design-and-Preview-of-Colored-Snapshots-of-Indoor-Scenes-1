程序使用说明：
csv_color.py
	首先将使用100color1.jpg以及统计搭配.csv统计出label.csv，，可以改tolabel()方法里的data[i][j]=0,生成新的label.csv(改data[i][j]="",自己重命名label.csv为label1.csv），生成100ColorRGB.txt（以#分割）
countColor.py，读取label1.csv生成统计条件概率的colorCount.npy  
readColorPiece.py 将100color1.jpg生成新的gamut.jpg) 生成yuyitu.png不能用gpu加速，否则会使颜色有偏移

jiajuColorChange.txt是统计的10种最常见颜色，可直接复制到程序里
	
Sedian.py，可视化excel表格所有专家先验颜色点，同时生成不同色差公式下颜色映射的颜色点。
Input_output.py,指定一种家具及其颜色，输出该条件下另一种家具及颜色的概率。（输出数据直接复制到echarts，复制每项数据的前十个，第十一个是输入数据）
