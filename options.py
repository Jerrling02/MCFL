import random

import torch

class args_parser():
    def __init__(self, load_dict=None):
        if load_dict != None:
            self.algorithm = load_dict["algorithm"]
            self.is_layered = load_dict["is_layered"]
            # self.edge_choice = load_dict["edge_choice"]
        else:
            self.algorithm = "Ours"
            self.is_layered = 1

        # The cloud server groups clients with similar computation and communication time into K clusters
        # and appoints a stable client as the cluster head l_k which is realized by Cluster.cluster
        self.edge_choice = "our_cluster"
        # self.model = "lenet" or "logistic"
        self.model = "logistic"
        self.data_distribution = 0.5
        # number of cluster heads
        self.num_edges = 4
        # number of clients, can be set as 20, 28, 36 for compare
        self.num_clients = 36

        self.data_fix = True

        if self.model == "logistic":
            if self.data_distribution == 0.2:
                self.target_acc = 0.88
            if self.data_distribution == 0.5:
                self.target_acc = 0.875
            if self.data_distribution == 0.8:
                self.target_acc = 0.87
            self.upload_dim=30
        else:
            if self.data_distribution == 0.2:
                self.target_acc = 0.94
            if self.data_distribution == 0.5:
                self.target_acc = 0.93
            if self.data_distribution == 0.8:
                self.target_acc = 0.915
            self.upload_dim = 90

        self.RL_path = './result/RL_{}_{}_{}client{}edge.npz'.format(self.data_distribution, self.model, self.num_clients,self.num_edges)
        self.reward_path = './result/RL_reward_{}_{}_{}client{}edge.npz'.format(self.data_distribution, self.model, self.num_clients,self.num_edges)
        self.actor_path = './actor_model_{}_{}_{}client{}edge'.format(self.data_distribution, self.model, self.num_clients,self.num_edges)
        self.batch_size = 16
        # max number of FL global aggregation. if target_acc reached, FL train done
        self.max_ep_step = 50
        # number of edges' local update's iteration
        self.edge_num_iteration = 2
        # data_dist indicates non-iid methods, including "Baochun" and "ICML"
        self.data_dist = "Baochun"

        # tau indicates the number of edge_aggregation, we choose the int tau_k_t for each edge k from 1 to tau_max
        self.tau_max = 5
        self.client_select_size = 10
        # for compare algorithm, number of each edge's aggregation is fixed
        self.num_edge_aggregation = 3
        # for MNIST, dataset's sample num is 60000
        self.dataset_sample_num = 60000
        # data_num for local update
        self.data_num = self.dataset_sample_num // (self.num_clients + self.num_edges)

        # computation cost
        self.noise = -104
        self.capacitance = 1e-27


        # communication cost
        self.lr = 0.01
        self.lr_decay = 0.995
        self.lr_decay_epoch = 1
        self.momentum = 0
        self.weight_decay = 0
        self.gamma = 1
        self.rl_batch_size = 32
        self.memory_capacity = 10000
        self.TAU = 0.01
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.load = False
        self.load_path='./actor_model_{}_{}_{}client{}edge'.format(self.data_distribution, self.model, self.num_clients,self.num_edges)
        self.total_resource = 2500000

        #gamma distribution
        self.lamda=2

        #normal distribution
        self.mean=random.uniform(81,1413)
        self.std=10

        #transfer learning control factor
        self.theta=0.2
        self.transfer=False
