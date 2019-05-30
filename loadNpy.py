import numpy as np

pre_train = np.load("vgg16.npy", allow_pickle=True, encoding="latin1")

data_dic = pre_train.item()

print("------type-------")
print(type(data_dic))
print("------conv1_1  data-------")
print(data_dic['conv1_1'])   # 返回一个列表，该列表有两个array，表示conv1_1的权重w和偏置b
print("------conv1_1  shape-------")
print((data_dic['conv1_1'][1]).shape)