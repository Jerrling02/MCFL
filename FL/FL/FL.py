# from init import *


# from init import EDGES
# from FL.models.initialize_model import mnist_lenet
import math
import time

import numpy as np
import torch

from FL.FL import *
from FL.client import Client
from FL.edge import Edge
from FL.cloud import Cloud

from Cluster.cluster import clients_cluster,recluster
import copy
import random


class Federate_learning():
    def __init__(self, args, dataloaders,clients_bds):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.train_loaders, self.test_loaders, self.v_train_loader, self.v_test_loader, self.data_distribution = dataloaders
        # args = args.num_clients -args.num_edges
        self.args = args
        self.clients_bds=clients_bds

        self.clients = [Client(id=cid,
                               train_loader=self.train_loaders[cid],  # train_loaders[cid],
                               test_loader=self.v_test_loader,
                               args=self.args,
                               # 把客户端个数改了
                               device=self.device,
                               data_distribution=self.data_distribution[cid]) for cid in range(self.args.num_clients)]
        # 初始化edge时加入了为edge分配数据过程
        self.edges = [Edge(id=eid,
                           train_loader=self.train_loaders[eid + self.args.num_clients],  # train_loaders[cid],
                           test_loader=self.v_test_loader,
                           args=self.args,
                           data_distribution=self.data_distribution[eid + self.args.num_clients]) for eid in
                      range(self.args.num_edges)]
        self.cloud = Cloud(args=self.args, edges=self.edges, test_loader=self.v_test_loader)

        self.num_clients = args.num_clients
        self.num_edges = args.num_edges
        self.load_path=args.load_path
        # self.step_count = 0

        self.tau_max = args.tau_max
        self.client_select_size = args.client_select_size
        self.num_max = self.num_clients - 4
        # 默认值设为了100
        # self.selected_num = args.selected_num

        self.reward = 0
        # 设置state为当前轮次的剩余计算通信资源，每个客户端的损失,当前轮次
        self.state = [-1] * (self.num_clients + self.num_edges * 2 + 1)
        self.state_space = len(self.state)
        self.observation_space = self.state_space
        self.action_bound = [0, 1]

        self.action_space = self.num_edges * self.num_max+ self.num_edges * self.tau_max
        # self.action_space = self.num_clients + self.num_edges * self.tau_max
        self.action_space =self.num_clients
        self.rest_resource = self.args.total_resource
        # clients_list, X, cmp_cap, com_cap, edge_list = clients_cluster(self.num_clients + self.num_edges,
        #                                                                self.num_edges)
        # edges_choices = [min(i // (self.num_clients // self.num_edges), self.num_edges) for i in
        #              range(self.num_clients)]
        # for client, edges_choice in zip(self.clients, edges_choices):  # 根据choices选择edges
        #         client.set_edge(edges_choice)
        #         self.edges[client.eid].add_client(client)


    def reset(self):
        # agents初始化
        # self.reward = 0
        torch.manual_seed(random.randint(0, 20))
        np.random.seed(random.randint(0, 20))

        # 清空模型参数
        self.cloud.reset_model()

        # shared_state_dict = self.cloud.shared_state_dict

        for edge in self.edges:
            edge.reset_model()
        # update客户端也重置模型
        for client in self.clients:
            client.reset_model()

        # 置cloud的testing_acc=0
        self.cloud.testing_acc = 0
        # self.recluster_threhold=5

        # # 如果只有一层，那么就全部连接到cloud
        # if self.args.is_layered == 0:
        #     for client in self.clients:
        #         self.cloud.clients.append(client)
        # # 如果分层了
        # else:
        # 固定的下层连接方式
        # if self.args.edge_choice == "our_cluster":
        clients_list, X, cmp_cap, com_cap, edge_list,cluster_result = clients_cluster(self.num_clients+self.num_edges, self.num_edges, self.args,self.clients_bds)
        self.cluster_result=cluster_result
        self.clients_list=[[] for _ in range(self.num_edges)]
        client_index=0
        for j in range(self.num_edges):
            for i in range(len(clients_list[j])):
                index = clients_list[j][i]
                client = self.clients[client_index]
                client.set_edge(j)
                # client.id= clients_list[j][i]
                self.edges[client.eid].add_client(client)
                client.cycles = cmp_cap[index][0]
                client.frequency = cmp_cap[index][1]
                client.iterations = cmp_cap[index][2]
                client.bandwidth = com_cap[index][0]
                client.transmit = com_cap[index][1]
                client.gain = com_cap[index][2]
                client_index = client_index + 1

            index = edge_list[j]
            self.edges[j].bandwidth = com_cap[index][0]
            self.edges[j].transmit = com_cap[index][1]
            self.edges[j].gain = com_cap[index][2]

            # 更新state
            for i in range(self.num_clients):
                self.state[i] = self.clients[i].local_update()
            # for i in range(self.num_clients, self.num_clients + self.num_edges):
            #     self.state[i] = self.rest_resource[i - self.num_clients]
            # for i in range(self.num_edges):
            #     time_list=[]
            #     for client in self.edges[i].clients:
            #         time_list.append(client.train_time())
            #     max_train_time=max(time_list)
            #     self.state[i+self.num_clients]=max_train_time + self.edges[i].comm_time_to_cloud()

            for i in range(self.num_edges):
                self.state[2 * i + self.num_clients] = self.edges[i].rest_time
                self.state[i * 2 + 1 + self.num_clients] = self.edges[i].test_model()

            self.state[-1] = 0

        return self.state

    def step(self, actions,param_dic,ep):

        done = False

        all_select_clients, taos_each_edge = action_choice(actions, self.num_clients, self.num_edges, self.num_max,
                                                           self.tau_max, self.edges, self.clients)
        cloud_loss, edge_loss, total_time,real_total_time, edge_time_list,param_dic = self.train(taos_each_edge,
                                                                                       all_select_clients,param_dic)
        # 最前面的state是客户端损失
        for i in range(self.num_clients):
            self.state[i] = self.clients[i].local_update()
        for i in range(self.num_edges):
            self.state[2 * i + self.num_clients] = edge_time_list[i]
            self.state[i * 2 + 1 + self.num_clients] = edge_loss[i]
        self.state[-1] += total_time
        # punishment1 = (total_resource - sum(self.args.total_resource[i])) if (total_resource - sum(self.args.total_resource[i]) > 0) else 0
        # 10是个参数，之后固定，暂定为10
        punishment1 = (self.rest_resource / 100) if (
                self.rest_resource < 0) else 0
        punishment2 = 0
        for eid in range(self.num_edges):
            # cluster_size = len(clients_in_cluster(all_select_clients, self.edges[eid].clients))
            cluster_size = len(self.edges[eid].select_clients)
            if self.edges[eid].n_k < cluster_size:
                punishment2 += 1


        if self.args.data_distribution == 0.5:
            self.reward = -total_time - (7 ** (self.args.target_acc - cloud_loss[0]) - 1) * 20
            objective_value =-real_total_time - (7 ** (self.args.target_acc - cloud_loss[0]) - 1) * 20
        elif self.args.data_distribution == 0.2:
            self.reward = -total_time - (9 ** (self.args.target_acc - cloud_loss[0]) - 1) * 20
            objective_value = -real_total_time - (9 ** (self.args.target_acc - cloud_loss[0]) - 1) * 20
        else:
            self.reward = -total_time - (5 ** (self.args.target_acc - cloud_loss[0]) - 1) * 20
            objective_value = -real_total_time - (5 ** (self.args.target_acc - cloud_loss[0]) - 1) * 20

        self.real_reward = real_total_time

        param_dic['objective_value'].append(objective_value)
        # print("\nobjective target value:{}\n".format(objective_value))
        # self.real_reward = total_time

        if cloud_loss[0] >= self.args.target_acc:
            done = True

        return copy.copy(self.state), self.reward, self.real_reward, cloud_loss[0],param_dic, done


    # train() runs for one global round
    def train(self, taos_each_edge, all_select_clients,param_dic):
        EDGES = self.num_edges
        clients = self.clients
        edges = self.edges
        cloud = self.cloud
        args = self.args
        client_loss = [0] * self.num_clients
        edge_loss = [0] * self.num_edges
        cloud_loss = [0]
        total_comm_cost = 0
        total_comp_cost = 0
        total_pri_cost = 0

        if self.args.is_layered:
            edge_time_list = [0] * self.args.num_edges
            real_edge_time_list = [0] * self.args.num_edges
            for eid in range(EDGES):
                time_list = []
                edge_cost = 0
                self.edges[eid].n_k = random.randint(len(self.edges[eid].clients) - 3, len(self.edges[eid].clients))
                cluster_size = len(self.edges[eid].select_clients)
                # 这一轮cluster没选到任何client
                if cluster_size > 0:
                    for num_edgeagg in range(taos_each_edge[eid]):
                        for i,client in enumerate(edges[eid].clients):
                            client_loss[client.id] = client.local_update()


                        for client in edges[eid].select_clients:
                            # if client.id in selected_clients:

                            client.send_to_edge(edges[eid])
                            # a = client.comm_cost()
                            # total_comm_cost += a
                            # edge_cost += a
                            # # 选中参与训练的客户端的计算成本
                            # b = client.comp_cost()
                            # total_comp_cost += b
                            # edge_cost += b
                            # 选中参与客户端的隐私成本
                            c = client.pri_cost()
                            total_pri_cost += c
                            edge_cost += c
                            # 选中客户端的训练时间
                            time_list.append(client.train_time())

                        edges[eid].local_update()
                        edges[eid].send_to_self()
                        # edge参与训练的隐私成本
                        d = edges[eid].pri_cost()
                        total_pri_cost += d
                        edge_cost += d
                        # edges[eid].clients = clients_in_cluster(selected_clients, edges[eid].clients)

                        edges[eid].aggregate()

                        for client in edges[eid].clients:
                            # if client in selected_clients:
                            # 这里edge将最新的model分享给所有所属clients
                            edges[eid].send_to_client(client)
                            edges[eid].send_to_self()


                    max_train_time = max(time_list)
                    edge_loss[eid] = edges[eid].test_model()
                    edges[eid].rest_cost = edges[eid].rest_cost - edge_cost / edges[eid].total_cost
                    edge_time_list[eid] = max_train_time * taos_each_edge[eid] + edges[eid].comm_time_to_cloud()
                    real_edge_time_list[eid] = max_train_time * taos_each_edge[eid] + edges[eid].comm_time_to_cloud()
                    edges[eid].select_clients = []

            total_time = max(edge_time_list)
            real_total_time = max(real_edge_time_list)

            for eid in range(EDGES):
                edges[eid].send_to_cloud(cloud)
                # edge send_to_cloud的通信成本
                # total_comm_cost += edges[eid].comm_cost()
            cloud.aggregate()
            # CLOUD 发送给edge
            for eid in range(EDGES):
                cloud.send_global_to_edge(self.edges[eid])
            for client in clients:
                cloud.send_to_client(client)
        else:
            time_list = []
            for num_edgeagg in range(args.num_edge_aggregation):
                for client in self.clients:
                    if client in all_select_clients:
                        client_loss[client.id] = client.loss

                for client in self.clients:
                    if client in all_select_clients:
                        client.send_to_cloud(cloud)
                        time_list.append(client.train_time())
                # for edge in self.edges:
                #     edge.send_to_cloud(cloud)

                cloud.aggregate()
                # CLOUD 发送给edge
                # for eid in range(EDGES):
                #     cloud.send_global_to_edge(self.edges[eid])
                for client in clients:
                    cloud.send_to_client(client)
            max_train_time = max(time_list)
            total_time = max_train_time
            real_total_time = max_train_time
            edge_time_list = []
        total_resource = total_comm_cost + total_comp_cost + total_pri_cost
        cloud_loss[0] = cloud.test_model()
        param_dic['accuracy'].append(cloud_loss[0])
        # torch.save(cloud.model.shared_layers.state_dict(), './FL_model_{}'.format(self.args.model))
        return cloud_loss, edge_loss, total_time, real_total_time,edge_time_list,param_dic

def action_choice(actions, num_clients, num_edges, num_max, tau_max, edges, clients):
    all_select_clients = []
    clients_choice_ = actions[num_edges:]
    # clients_choice_=actions
    for i in range(len(clients_choice_)):
        idx = clients_choice_[i]
        all_select_clients.append(clients[idx].id)

    taos_each_edge = [0] * num_edges
    for i in range(num_edges):
        for client in edges[i].clients:
            if client.id in all_select_clients:
                edges[i].select_clients.append(client)
        iterations = int(actions[i])
        # iterations =3
        # client=edges[i].clients[client_idx]
        # for client in edges[i].clients:
        #     edges[i].select_clients.append(client)
        # all_select_clients.append(client.id)
        taos_each_edge[i] = iterations + 1
    return all_select_clients, taos_each_edge


def clients_in_cluster(nums1, nums2):
    m = {}
    if len(nums1) < len(nums2):
        nums1, nums2 = nums2, nums1
    for i in nums1:
        if i not in m:
            m[i] = 1
        else:
            m[i] += 1
    result = []
    for i in nums2:
        if i in m and m[i]:
            m[i] -= 1
            result.append(i)
    return result