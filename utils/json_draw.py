import json
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

json_list = sys.argv[1:]
length = None #None
show_all = True
scale = 20
acc_loss = []
acc_grad = []
xticks = []
with open(json_list[0], 'r') as f:
    metrics_dic = json.load(f)
    train_accuracies = metrics_dic['train_accuracies']
    for i, acc in enumerate(train_accuracies):
        xticks.append(i)
        if show_all:
            acc_loss.extend([a/b for a,b in zip(acc[3], acc[2])])
        else:
            acc_loss.append(np.sum(acc[3]) * 1.0 / np.sum(acc[2]))
with open(json_list[1], 'r') as f:
    metrics_dic = json.load(f)
    train_accuracies = metrics_dic['train_accuracies']
    for acc in train_accuracies:
        if show_all:
            acc_grad.extend([a / b for a, b in zip(acc[3], acc[2])])
        else:
            acc_grad.append(np.sum(acc[3]) * 1.0 / np.sum(acc[2]))


sub_length = length if length is not None else len(acc_loss)
acc_loss = acc_loss[:sub_length:scale]
acc_grad = acc_grad[:sub_length:scale]
X = [ i for i in range(0,len(acc_grad))]

ln1, = plt.plot(X,acc_loss,color='red',linewidth=0.3,linestyle='-')
ln2, = plt.plot(X,acc_grad,color='blue',linewidth=0.3,linestyle='-')
xticks = xticks[::20]
a = math.ceil(len(X)/len(xticks))
plt.xticks(X[::a], xticks)

my_font = fm.FontProperties(fname="../font/wqy-microhei.ttc")
plt.title("训练精度比较",fontproperties=my_font) #设置标题及字体
plt.legend(handles=[ln1, ln2], labels=['根据loss选择用户', '根据grad选择用户'], prop=my_font)

ax = plt.gca()
ax.spines['right'].set_color('none')  #
ax.spines['top'].set_color('none')    #

plt.show()


# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm  # 字体管理器
#
# x_data = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
# y_data = [58000, 60200, 63000, 71000, 84000, 90500, 107000]
# y_data2 = [52000, 54200, 51500, 58300, 56800, 59500, 62700]
#
# ln1, = plt.plot(x_data, y_data, color='red', linewidth=2.0, linestyle='--')
# ln2, = plt.plot(x_data, y_data2, color='blue', linewidth=3.0, linestyle='-.')
#
# my_font = fm.FontProperties(fname="../font/wqy-microhei.ttc")
#
# plt.title("电子产品销售量", fontproperties=my_font)  # 设置标题及字体
#
# plt.legend(handles=[ln1, ln2], labels=['鼠标的年销量', '键盘的年销量'], prop=my_font)
#
# ax = plt.gca()
# ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
# ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
#
# plt.show()
