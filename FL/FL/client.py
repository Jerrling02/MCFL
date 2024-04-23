import math
import random
import torch
from FL.models.initialize_model import initialize_model
import copy
import numpy as np


class Client:

    def __init__(self, id, train_loader, test_loader, args, device, data_distribution):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        self.receiver_buffer = {}
        self.epoch = 0
        self.args = args

        # 计算computation cost所用的参数
        self.frequency = 0
        self.cycles = 0
        self.old_cycles=0
        self.iterations = 0

        # communication cost
        # self.transmit = 0
        self.bandwidth = []
        self.bandwidth_now=0
        # self.old_bandwidth = 0
        # self.gain = 0
        self.pps = random.randint(10, 30)
        self.privacy_b = random.uniform(1, 1.5)
        self.privacy_p = random.uniform(100, 200)

        # self.weight = 0.5
        # self.weight_float = 0.5
        self.eid = -1
        self.device = device
        self.testing_acc = 0
        # self.train_index = 0
        self.data_distribution = data_distribution
        # 这里可能有兼容错误，因为我们是列表
        # self.data_num = self.args.data_num
        # self.loss=
        self.data_num = self.args.data_num

    def local_update(self):
        num_iter = self.iterations
        # num_iter = 5
        loss = 0.0
        for i in range(num_iter):
            # data_samples=random.sample(self.train_loader, 32)
            for batch_id, (inputs, labels) in enumerate(self.train_loader):
                # data = self.train_loader[self.train_index]
                # inputs, labels = data
                # inputs = inputs.to(self.device)
                # labels = labels.to(self.device)
                loss += self.model.optimize_model(input_batch=inputs,
                                                  label_batch=labels)
                # self.train_index += 1
                # self.train_index %= self.data_num
        # self.epoch += 1
        # self.model.exp_lr_sheduler(epoch=self.epoch)
        loss /= num_iter
        self.loss = loss
        return loss

    def test_model(self):
        correct = 0.0
        total = 0.0
        for data in self.test_loader:
            inputs, labels = data
            break
        size = labels.size(0)
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                # inputs = inputs.to(self.device)
                # labels = labels.to(self.device)
                outputs = self.model.test_model(input_batch=inputs)
                _, predict = torch.max(outputs, 1)
                total += size
                correct += (predict == labels).sum()
        self.testing_acc = correct.item() / total
        return correct.item() / total

    def send_to_edge(self, edge):
        edge.receiver_buffer[self.id] = copy.deepcopy(self.model.shared_layers.state_dict())

    def send_to_cloud(self, cloud):

        cloud.receiver_buffer[self.id] = copy.deepcopy(self.model.shared_layers.state_dict())

    def set_edge(self, eid):
        self.eid = eid

    def reset_model(self):
        self.receiver_buffer = {}
        self.epoch = 0
        # self.weight = random.random()
        # self.weight_float = self.weight
        self.model = initialize_model(self.args, self.device)
        # self.train_index = 0
        # self.loss = 10
        # 因为默认数据集合不变，所以这里没有重新初始化data_num
        # 但是数据同步那里可能会有影响，所以在那里要更新

    def time_vary_com_cmp(self,t):

        self.cycles = min(np.random.gamma(self.args.lamda, 1 / self.args.lamda) * self.old_cycles,30)
        self.bandwidth_now = np.mean(self.bandwidth[t:t+5])
        # com_c=self.comm_time()
        # cmp_c=self.comp_time()
        # if com_c+cmp_c > com+cmp:
        #     self.bandwidth=max(0.5,self.bandwidth-0.5)
        #     self.cycles = max(self.cycles + 0.01, 30)
        # else:
        #     self.bandwidth = min(self.bandwidth + 0.2,30)
        #     self.cycles = max(self.cycles - 0.01, 30)




    # 定义每个客户端一次训练的计算时间（每个客户端迭代次数给定）
    def comp_time(self):
        comp_time = self.data_num * self.iterations * self.cycles / self.frequency
        return comp_time

    def comm_time(self):

        # r_i = (self.bandwidth_now * math.log2(1 + (self.gain * self.transmit) / self.args.noise))
        r_i=self.bandwidth_now
        comm_time = self.args.upload_dim / r_i
        if self.args.is_layered == 0:
            comm_time *= 10
        return comm_time

    def train_time(self):
        return self.comm_time() + self.comp_time()

    # 计算send_to_edge的通信成本 z_{i,l_k}
    # def comm_cost(self):
    #     comm_cost = self.comm_time() * self.pps
    #     return comm_cost
    #
    # def comp_cost(self):
    #     comp_cost = self.args.capacitance * self.data_num * self.cycles * (self.frequency ** 2)
    #     return comp_cost

    def pri_cost(self):
        pri_cost = self.privacy_b * self.privacy_p
        return pri_cost
