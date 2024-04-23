import copy
from FL.average import average_weights
from FL.models.initialize_model import initialize_model
import torch

class Cloud():
    def __init__(self, args, edges, test_loader, shared_layers=0):
        self.device = args.device
        self.receiver_buffer = {}
        self.edges = edges
        self.args = args
        self.num_edges = args.num_edges
        self.test_loader = test_loader
        self.model = initialize_model(args, self.device)
        self.shared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
        self.testing_acc = 0
        if not self.args.is_layered:
            self.clients = []
        
    def aggregate(self):
        received_dict = []
        sample_num = []
        if self.args.algorithm == "Ours":
            if self.args.is_layered:
                edges_aggregate_num = 0
                for edge in self.edges:
                    if edge.all_data_num:
                        edges_aggregate_num += 1
                        received_dict.append(self.receiver_buffer[edge.id])
                        sample_num.append(edge.all_data_num)

                if edges_aggregate_num == 0:
                    return
                self.shared_state_dict = average_weights(w=received_dict,
                                                         s_num=sample_num)
                self.model.update_model(copy.deepcopy(self.shared_state_dict))
                # edges_aggregate_num = 1
                # for edge in self.edges:
                #     if edge.all_weight_num:
                #         edges_aggregate_num += 1
                #         received_dict.append(self.receiver_buffer[edge.id])
                #         sample_num.append(edge.all_weight_num)
                #
                # if edges_aggregate_num == 0:
                #     return
                # self.shared_state_dict = average_weights(w = received_dict,
                #                                         s_num= sample_num)
                # self.model.update_model(copy.deepcopy(self.shared_state_dict))
            # else:
            #     self.all_weight_num = 0
            #     for client in self.clients:
            #         if client.weight:
            #             self.all_weight_num += client.weight
            #             received_dict.append(self.receiver_buffer[client.id])
            #             sample_num.append(client.weight)
            #     if self.all_weight_num == 0:
            #         return
            #     self.shared_state_dict = average_weights(w = received_dict,
            #                                             s_num= sample_num)
            #     self.model.update_model(copy.deepcopy(self.shared_state_dict))
            else:
                clients_aggregate_num = 0

                for client in self.clients:
                    if client.data_num:
                        clients_aggregate_num += 1
                        received_dict.append(self.receiver_buffer[client.id])
                        sample_num.append(client.data_num)

                if clients_aggregate_num == 0:
                    return
                self.shared_state_dict = average_weights(w=received_dict,
                                                         s_num=sample_num)
                self.model.update_model(copy.deepcopy(self.shared_state_dict))
        elif self.args.algorithm == 'FD_avg':
            if self.args.is_layered: 
                edges_aggregate_num = 0
                
                for edge in self.edges:
                    if edge.all_data_num:
                        edges_aggregate_num += 1
                        received_dict.append(self.receiver_buffer[edge.id])
                        sample_num.append(edge.all_data_num)
                        
                if edges_aggregate_num == 0:
                    return
                self.shared_state_dict = average_weights(w = received_dict,
                                                        s_num= sample_num)
                self.model.update_model(copy.deepcopy(self.shared_state_dict))
            else:
                clients_aggregate_num = 0
                
                for client in self.clients:
                    if client.data_num:
                        clients_aggregate_num += 1
                        received_dict.append(self.receiver_buffer[client.id])
                        sample_num.append(client.data_num)
                        
                if clients_aggregate_num == 0:
                    return
                self.shared_state_dict = average_weights(w = received_dict,
                                                        s_num= sample_num)
                self.model.update_model(copy.deepcopy(self.shared_state_dict))    
        elif self.args.algorithm == 'Fair':
            if not self.args.is_layered: 
                for client in self.clients:
                    client.test_model()
                top_clients = sorted(self.clients, key = lambda x: x.testing_acc - self.testing_acc, reverse=True)
                top_clients = top_clients[:int(len(self.clients) / 1.5)]
                clients_aggregate_num = 0
                for client in top_clients:
                # for client in self.clients:
                    if client.data_num:
                        clients_aggregate_num += 1
                        received_dict.append(self.receiver_buffer[client.id])
                        sample_num.append((client.testing_acc - self.testing_acc) * client.data_num)
                        # sample_num.append(client.data_num)
                if sum(sample_num) == 0:
                    return
                if clients_aggregate_num == 0:
                    return
                self.shared_state_dict = average_weights(w = received_dict,
                                                        s_num= sample_num)
                self.model.update_model(copy.deepcopy(self.shared_state_dict))  
                # delta_loss = [0] * self.clients
                # for i, client in enumerate(self.clients):
                #     delta_loss[i] = (client.testing_acc - self.testing_acc, i)
                
                
                
                # # 找topK个最大的  返回下标  也就是client.id
                # topk = sorted(delta_loss, key = lambda x: x[0], reverse=True)
                # topk_data_num = [0] * self.clients
                # topk_index = []
                # for t in topk:
                #     _, i = t
                #     topk_index.append(i)
                #     topk_data_num[i] = self.clients[i].data_num
                # topk_sum_data_num = sum(topk_data_num)
                # clients_aggregate_num = 0
                # for index in topk_index:
                #     if client.data_num:
                #         clients_aggregate_num += 1
                #         received_dict.append(self.receiver_buffer[client.id])
                #         sample_num.append(delta_loss[index] * client.data_num / topk_sum_data_num)
                # if sum(sample_num) == 0:
                #     return
                # # for client in self.clients:
                # #     if client.data_num:
                # #         clients_aggregate_num += 1
                # #         received_dict.append(self.receiver_buffer[client.id])
                # #         sample_num.append(client.data_num)
                        
                # if clients_aggregate_num == 0:
                #     return
                
 
            else:
                pass              
        else:
            pass
    def send_to_client(self, client):
        client.receiver_buffer = copy.deepcopy(self.shared_state_dict)
        client.model.update_model(client.receiver_buffer)

    # 添加 将全局模型发送给edge作为下一轮的初始模型 的方法
    def send_global_to_edge(self, edge):
        edge.self_receiver_buffer = copy.deepcopy(self.shared_state_dict)
        edge.model.update_model(edge.self_receiver_buffer)

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
                outputs = self.model.test_model(input_batch= inputs)
                _, predict = torch.max(outputs, 1)
                total += size
                correct += (predict == labels).sum()
        self.testing_acc = correct.item() / total
        return correct.item() / total
    
    def reset_model(self):
        self.receiver_buffer = {}
        self.model = initialize_model(self.args, self.device)
        self.shared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
