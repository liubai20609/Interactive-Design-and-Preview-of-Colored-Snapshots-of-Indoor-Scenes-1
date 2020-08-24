Program instructions：
1. csv_color.py：First, use 100color1.jpg and statistical collocation.csv to calculate label.csv. You can change data[i][j]=0 in the tolabel() method to generate a new label.csv (change data[i][j ]="", rename label.csv to label1.csv by yourself), generate 100ColorRGB.txt (divided by #)
2. countColor.py：Read label1.csv to generate colorCount.npy for statistical conditional probability
3. readColorPiece.py：Generate new gamut.jpg from 100color1.jpg
4.jiajuColorChange.txt：The 10 most common colors in statistics
5.Sedian.py，可视化excel表格所有专家先验颜色点，同时生成不同色差公式下颜色映射的颜色点。
6. input_output.py：Specify a kind of furniture and its color, and output the probability of another kind of furniture and color under this condition.
