import copy
import torch
from torch import nn

def average_weights(w, s_num):
    """
    :param w:键为CNN层数，值为该层的模型权重参数
    :param s_num: 列表，元素为采样总数列表，元素为所有客户端的每层采样总数列表，每个列表是每个客户端的采样总数列表
    :return :返回平均聚合之后的模型参数
    """

    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] = w_avg[k] + torch.mul(w[i][k], s_num[i]/temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg