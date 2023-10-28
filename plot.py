import math
import matplotlib.pyplot as plt
import numpy as np
from utils import smooth
import scipy.io as sio






data_path = "EORL_log.txt"
target = 'reward:  '
target_data_len = 19  # 提取数据的长度
# 提取x1的数据


num_list = []  # 将提取出来的数据保存到列表,并在最后返回
data = open(data_path, encoding="utf-8")  # 打开文件
str1 = data.read()  # 将文件中读取到的内容转化为字符串
data.close()  # 关闭文件
data = open(data_path, encoding="utf-8")  # 打开文件

for line in data:
    index = line.find(target)  # 查找字符串str1中str2字符串的位置
    if index == -1:
        continue
    test_str = line[index+len(target):index+len(target)+target_data_len]
    if ' ' in test_str:
        test_str = test_str.split(' ')[0]
    num_list.append(float(test_str))  # 将需要的数据提取到列表中
    # line = line.replace(target, 'xxxx', 1)  # 替换掉已经查阅过的地方,' xxxx '表示替换后的内容，1表示在字符串中的替换次数为1

data.close()  # 关闭文件
# reward  = np.array(num_list)
# return_list  = smooth([num_list])
# print(return_list[0])
np.save('EORL.npy',num_list)

sio.savemat('EORL.mat',{'EORL':num_list})



