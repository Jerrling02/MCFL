import copy
import math
import random
from FL.average import average_weights
from FL.models.initialize_model import initialize_model
import torch
import numpy as np


class Edge():

    def __init__(self, id, train_loader, test_loader, args, data_distribution):
        self.device = args.device

        self.transmit_rate = random.randint(100, 150)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.id = id
        self.clients = []
        self.select_clients = []
        self.receiver_buffer = {}

        self.self_receiver_buffer = {}
        self.epoch = 0
        self.weight = 0.5
        self.weight_float = 0.5
        self.testing_acc = 0
        self.data_num = args.data_num

        self.all_weight_num = 0
        self.all_weight_float_num = 0
        self.all_data_num = 0
        self.args = args
        self.shared_state_dict = {}
        self.model = initialize_model(args, self.device)
        self.shared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
        self.total_cost = random.uniform(150000, 200000)
        self.rest_cost = 0
        self.rest_time = 0

        # self.train_index = 0
        self.data_distribution = data_distribution

        # 计算computation cost所用的参数
        self.frequency = 0
        self.cycles = 0
        self.old_cycles=0
        self.iterations = 0

        # self.transmit = 0
        self.bandwidth = []
        self.bandwidth_now = 0
        # self.gain = 0

        self.pps = random.randint(10, 30)

        self.privacy_b = random.randint(10, 20)
        self.privacy_p = random.randint(10, 20)


    # 添加本地更新方法，与client中的相同
    def local_update(self):
        num_iter = self.args.edge_num_iteration
        loss = 0.0
        for i in range(num_iter):
            # data_samples = random.sample(self.train_loader, 32)
            # for batch_id, (inputs, labels) in enumerate(data_samples):
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

    # 添加test_model方法
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

    def aggregate(self):
        received_dict = []
        sample_num = []
        self.all_data_num = 0
        for client in self.select_clients:
            if client.data_num:
                self.all_data_num += client.data_num
                received_dict.append(self.receiver_buffer[client.id])
                sample_num.append(client.data_num)

        # 在聚合方法中将edge自己的权重加进去
        self.all_data_num += self.data_num
        received_dict.append(self.self_receiver_buffer)
        sample_num.append(self.data_num)

        if self.all_data_num == 0:
            return

        self.shared_state_dict = average_weights(w=received_dict,
                                                 s_num=sample_num)
        self.model.update_model(copy.deepcopy(self.shared_state_dict))

        # if self.args.algorithm == 'W_avg':
        #     for client in self.clients:
        #         self.all_weight_float_num += client.weight_float
        #         if client.weight:
        #             self.all_weight_num += client.weight
        #             received_dict.append(self.receiver_buffer[client.id])
        #             sample_num.append(client.weight)
        #     # 在聚合方法中将edge自己的权重加进去
        #     if self.weight:
        #         self.all_weight_num += self.weight
        #         received_dict.append(self.self_receiver_buffer)
        #         sample_num.append(self.weight)
        #
        #     if self.all_weight_num == 0:
        #         return
        #
        #     self.shared_state_dict = average_weights(w=received_dict,
        #                                             s_num=sample_num)
        # elif self.args.algorithm == 'FD_avg':
        #     for client in self.clients:
        #         # 为了保证train里面计算cost的时候兼容
        #         self.all_weight_float_num += client.weight_float
        #         self.all_weight_num += client.weight
        #         if client.data_num:
        #             self.all_data_num += client.data_num
        #             received_dict.append(self.receiver_buffer[client.id])
        #             sample_num.append(client.data_num)
        #
        #     # 在聚合方法中将edge自己的权重加进去
        #     self.all_weight_num += self.weight
        #     received_dict.append(self.self_receiver_buffer)
        #     sample_num.append(self.weight)
        #
        #     if self.all_data_num == 0:
        #         return
        #
        #     self.shared_state_dict = average_weights(w= received_dict,
        #                                             s_num=sample_num)
        # else:
        #     pass

    # 在edge发送cluster全局模型时发给自己
    def send_to_self(self):
        self.self_receiver_buffer = copy.deepcopy(self.model.shared_layers.state_dict())
        # self.model.update_model(self.self_receiver_buffer)

    def send_to_client(self, client):
        client.receiver_buffer = copy.deepcopy(self.shared_state_dict)
        client.model.update_model(client.receiver_buffer)

    def send_to_cloud(self, cloud):
        cloud.receiver_buffer[self.id] = copy.deepcopy(self.shared_state_dict)

    def remove_client(self, client):
        self.receiver_buffer.pop(client.id)
        self.clients.remove(client)
        client.set_edge(-2)
        # 上面聚合的时候清0了，所以不需要了
        # self.all_weight_num -= client.weight
        # self.all_weight_float_num -= client.weight_float
        # self.all_data_num -= client.data_num

    def add_client(self, client):
        self.clients.append(client)
        # 上面聚合的时候清0了，所以不需要了
        # self.all_weight_num += client.weight
        # self.all_weight_float_num += client.weight_float
        # self.all_data_num += client.data_num

    def reset_model(self):
        self.receiver_buffer = {}
        self.self_receiver_buffer = {}
        self.model = initialize_model(self.args, self.device)
        self.shared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
        self.epoch = 0
        self.weight = random.random()
        self.weight_float = self.weight
        # self.train_index = 0
        self.model = initialize_model(self.args, self.device)
        self.select_clients = []
        self.clients=[]
        # self.loss = 10

    def comm_cost(self):
        comm_cost = self.pps * self.args.upload_dim / \
                    (self.bandwidth * math.log2(1 + (self.gain * self.transmit) / self.args.noise))
        return comm_cost

    def pri_cost(self):
        pri_cost = self.privacy_b * self.privacy_p
        return pri_cost

    def time_vary_com_cmp(self,t):

        self.cycles = min(np.random.gamma(self.args.lamda, 1 / self.args.lamda) * self.old_cycles, 30)
        self.bandwidth_now = self.bandwidth[t]
        # x=random.random()
        # if x<0.5:
        #     self.cycles = self.cycles + 0.05
        #     self.bandwidth = self.bandwidth + 0.05
        # else:
        #     self.cycles = self.cycles - 0.05
        #     self.bandwidth = self.bandwidth - 0.05

    def comm_time_to_cloud(self):
        self.transmit_rate = self.bandwidth_now
        # self.transmit_rate = (self.bandwidth_now * math.log2(1 + (self.gain * self.transmit) / self.args.noise))
        return self.args.upload_dim / self.transmit_rate *10
        # return self.args.upload_dim / self.transmit_rate

    # 定义每个客户端一次训练的计算时间（每个客户端迭代次数给定）
    def comp_time(self):
        comp_time = self.data_num * self.iterations * self.cycles / self.frequency
        return comp_time

    #作为客户端上传的时间
    def comm_time(self):
        # r_i = (self.bandwidth_now * math.log2(1 + (self.gain * self.transmit) / self.args.noise))
        r_i = self.bandwidth_now
        comm_time = self.args.upload_dim / r_i
        if self.args.is_layered == 0:
            comm_time *= 10
        return comm_time

    def train_time(self):
        return self.comm_time() + self.comp_time()
