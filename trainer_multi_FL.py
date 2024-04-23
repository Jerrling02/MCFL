#import gym
import datetime

import numpy as np
import matplotlib.pyplot as plt
from FL.FL import Federate_learning as FL
from FL.datasets.get_data import *
from options import args_parser
from tqdm import tqdm
import data_dist
import json
#from Discrete_SAC_Agent import SACAgent
#from multi_action_Agent import SACAgent

TRAINING_EVALUATION_RATIO = 4
RUNS = 20
EPISODES_PER_RUN = 20000
# STEPS_PER_EPISODE = 200
START_EPISODE= 50
# layer_num = 21


import torch


#from utilities.ReplayBuffer import ReplayBuffer



class Actor(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation,num_clients,num_edges, tau_max,PATH , LOAD,device):
        super(Actor, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Linear(in_features=128, out_features=128)
        self.layer_out1 = torch.nn.Linear(in_features=128, out_features = num_clients)
        self.layers_out2=torch.nn.ModuleList([torch.nn.Linear(128, tau_max) for _ in range(num_edges)])
        # self.layers_out2 = torch.nn.Linear(in_features=128, out_features = tau_max)
        # self.output_layer = torch.nn.Linear(in_features=132, out_features=output_dimension)
        self.normal=torch.nn.LayerNorm(normalized_shape=num_clients, eps=0, elementwise_affine=False)
        self.output_activation = output_activation
        self.num_edges=num_edges
        if LOAD:
            self.load_model(PATH,device)


    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        #layer_3_output = self.output_layer(layer_2_output)

        #x1, x2 = layer_3_output.split([3, 20], dim=1)
        y=self.output_activation(self.normal(self.layer_out1(layer_2_output)))
        # y_1=torch.zeros(y.shape)
        # y_1[(torch.arange(len(y)).unsqueeze(1), torch.topk(y, 10).indices)] = 1
        # y1=self.output_activation(y)
        # x = []
        #
        for i in range(self.num_edges):
            x.append(self.output_activation(self.layers_out2[i](layer_2_output)))

        output = x[0]
        for i in range(1, len(x)):
            output = torch.cat([output, x[i]], dim=1)
        output = torch.cat([output, y], dim=1)
        # output=y
        return output

    def load_model(self, PATH,device):
        self.load_state_dict(torch.load(PATH,map_location=device))

class Critic(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Critic, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Linear(in_features=128, out_features=64)
        self.output_layer = torch.nn.Linear(in_features=64, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = self.output_layer(layer_2_output)


        output = self.output_activation(layer_3_output)
        return output

class SACAgent:

    ALPHA_INITIAL = 1.
    REPLAY_BUFFER_BATCH_SIZE = 50
    DISCOUNT_RATE = 1
    LEARNING_RATE = 10 ** -4
    SOFT_UPDATE_INTERPOLATION_FACTOR = 0.01

    def __init__(self, environment,LOAD):
        self.environment = environment
        self.num_clients = environment.num_clients
        self.num_edges = environment.num_edges
        self.tau_max = environment.tau_max
        self.client_select_size = environment.client_select_size
        self.state_dim = environment.state_space
        self.action_dim = environment.action_space
        self.device = environment.device
        #self.action_dim = 3*20
        #self.state_dim = self.environment.observation_space.shape[0]
        #self.action_dim = self.environment.action_space.n
        self.critic_local = Critic(input_dimension=self.state_dim,
                                    output_dimension=self.action_dim)
        self.critic_local2 = Critic(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)
        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.LEARNING_RATE)

        self.critic_target = Critic(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)
        self.critic_target2 = Critic(input_dimension=self.state_dim,
                                      output_dimension=self.action_dim)

        self.soft_update_target_networks(tau=1.)

        self.actor_local = Actor(
            input_dimension=self.state_dim,
            output_dimension=self.action_dim,
            output_activation=torch.nn.Softmax(dim=1),
            num_clients= self.num_clients,
            num_edges=self.num_edges,
            tau_max=self.tau_max,
            PATH = environment.load_path,
            LOAD = LOAD,
            device=self.device
        )

        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(self.environment)

        self.target_entropy = 0.98 * -np.log(1 / self.action_dim)
        self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.LEARNING_RATE)

    def get_next_action(self, state, theta, evaluation_episode=False):
        if evaluation_episode:
            discrete_action = self.get_action_deterministically(state)
        else:
            discrete_action = self.get_action_nondeterministically(state)
        return discrete_action
    def consult_expert(self,state):
        self.get_action_deterministically(state)

    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = []
        action_edge_num = self.tau_max
        client_actions = []
        edges_action_probabilities = [0] * self.num_edges
        for i in range(self.num_edges):
            edges_action_probabilities[i] = action_probabilities[i * action_edge_num:(i + 1) * action_edge_num]
        client_actions = action_probabilities[self.num_edges * action_edge_num:]
        for i in range(self.num_edges):
           discrete_action.append(np.random.choice(range(0, self.tau_max),
                                                    p=edges_action_probabilities[i][0: self.tau_max]))
        client_choice = list(np.random.choice(range(0, self.num_clients), size=10, p=client_actions, replace=False))
        discrete_action = discrete_action + client_choice
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = []
        # action_edge_num = self.num_max + self.tau_max
        action_edge_num = self.tau_max
        edges_action_probabilities = [0] * self.num_edges
        for i in range(self.num_edges):
            edges_action_probabilities[i] = action_probabilities[i * action_edge_num:(i + 1) * action_edge_num]
        client_actions = action_probabilities[self.num_edges * action_edge_num:]
        for i in range(self.num_edges):
            discrete_action.append(np.argmax(edges_action_probabilities[i][0: self.tau_max]))
        for i in range(10):
            index = np.argmax(client_actions)
            client_actions[index] = 0
            discrete_action.append(index)
        return discrete_action

    def train_on_transition(self, state, discrete_action, next_state, reward, done):
        count = 0
        for i in range(self.num_edges):
            discrete_action[i] = discrete_action[i] + count
            count += self.tau_max
        idx = self.num_edges
        for i in range(len(discrete_action) - self.num_edges):
            discrete_action[i + idx] = discrete_action[i + idx] + idx
        transition = (state, discrete_action, reward, next_state, done)
        self.train_networks(transition)

    def train_networks(self, transition):
        # Set all the gradients stored in the optimisers to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        # Calculate the loss for this transition.
        self.replay_buffer.add_transition(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if self.replay_buffer.get_size() >= self.REPLAY_BUFFER_BATCH_SIZE:
            # get minibatch of 100 transitions from replay buffer
            minibatch = self.replay_buffer.sample_minibatch(self.REPLAY_BUFFER_BATCH_SIZE)
            minibatch_separated = list(map(list, zip(*minibatch)))

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(np.array(minibatch_separated[0]), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(minibatch_separated[1]),dtype=torch.float32)
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2])).float()
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]))
            done_tensor = torch.tensor(np.array(minibatch_separated[4]))
            #actions_tensor_2 = torch.tensor(np.array(minibatch_separated[5]), dtype=torch.float32)

            critic_loss, critic2_loss = \
                self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (
                    torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities
            )).sum(dim=1)

            next_q_values = rewards_tensor + ~done_tensor * self.DISCOUNT_RATE*soft_state_values

        #actions = []
        #num = self.vehicle_num * 2
        temp = torch.split(actions_tensor, 1, dim = 1)


        #soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        '''
        soft_q_values = self.critic_local(states_tensor)
        soft_q_values_1 = soft_q_values.gather(1, actions_tensor_1.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_values_2 = soft_q_values.gather(1, actions_tensor_2.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_values2 = self.critic_local2(states_tensor)
        soft_q_values2_1 = soft_q_values2.gather(1, actions_tensor_1.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_values2_2 = soft_q_values2.gather(1, actions_tensor_2.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        '''
        soft_q_value = self.critic_local(states_tensor)
        soft_q_value2 = self.critic_local2(states_tensor)
        soft_q_values = []
        soft_q_values2 = []

        #a = temp[1].type(torch.int64).squeeze()

        for i in range(len(temp)):
            soft_q_values.append(soft_q_value.gather(1, temp[i].type(torch.int64).squeeze().unsqueeze(-1)).squeeze(-1))
            soft_q_values2.append(soft_q_value2.gather(1, temp[i].type(torch.int64).squeeze().unsqueeze(-1)).squeeze(-1))
        critic_square_error = torch.nn.MSELoss(reduction="none")(sum(soft_q_values), next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(sum(soft_q_values2), next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, states_tensor,):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.alpha * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)

        return action_probabilities.squeeze(0).detach().numpy()

    def soft_update_target_networks(self, tau=SOFT_UPDATE_INTERPOLATION_FACTOR):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)


class ReplayBuffer:

    def __init__(self, environment, capacity=5000):
        transition_type_str = self.get_transition_type_str(environment)
        self.buffer = np.zeros(capacity, dtype=transition_type_str)
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None

    def get_transition_type_str(self, environment):
        #state_dim = environment.observation_space.shape[0]
        state_dim = environment.state_space
        state_dim_str = '' if state_dim == () else str(state_dim)
        #state_type_str = environment.observation_space.sample().dtype.name
        state_type_str = "float32"
        #action_dim = "2"
        action_dim = environment.num_edges+10
        #action_dim = environment.action_space.shape
        action_dim_str = '' if action_dim == () else str(action_dim)
        #action_type_str = environment.action_space.sample().__class__.__name__
        action_type_str = "int"

        # type str for transition = 'state type, action type, reward type, state type'
        transition_type_str = '{0}{1}, {2}{3}, float32, {0}{1}, bool'.format(state_dim_str, state_type_str,
                                                                             action_dim_str, action_type_str)

        return transition_type_str

    def add_transition(self, transition):
        self.buffer[self.head_idx] = transition
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)
        return self.buffer[self.indices]

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def get_size(self):
        return self.count


if __name__ == "__main__":

    np.random.seed(1)

    args = args_parser()

    if args.data_fix == False:
        data_distribution = np.array(data_dist.data_dist())
    else:
        if args.data_distribution == 0.8:
            if args.num_clients == 20:
                data_distribution = [
                    [2000, 0, 0, 0, 124, 0, 0, 62, 62, 248],  # 0
                    [2000, 0, 186, 0, 124, 62, 62, 62, 0, 0],  # 1
                    [2000, 0, 124, 186, 0, 62, 0, 62, 62, 0],  # 2
                    [2000, 0, 186, 0, 62, 124, 0, 0, 124, 0],  # 3
                    [2000, 0, 62, 0, 62, 62, 124, 0, 0, 186],  # 4
                    [2000, 0, 0, 186, 0, 0, 124, 0, 62, 124],  # 5
                    [0, 2000, 0, 62, 124, 0, 0, 310, 0, 0],  # 6
                    [0, 2000, 0, 124, 62, 62, 0, 0, 124, 124],  # 7
                    [0, 2000, 186, 62, 0, 62, 62, 62, 62, 0],  # 8
                    [0, 2000, 0, 124, 62, 62, 62, 62, 62, 62],  # 9
                    [0, 2000, 62, 124, 0, 0, 124, 62, 62, 62],  # 10
                    [0, 2000, 0, 62, 124, 62, 0, 124, 124, 0],  # 11
                    [0, 0, 2124, 0, 62, 124, 124, 0, 0, 62],  # 12
                    [0, 0, 62, 2062, 124, 62, 0, 62, 0, 124],  # 13
                    [0, 0, 62, 124, 2000, 0, 0, 0, 186, 124],  # 14
                    [0, 0, 0, 0, 124, 2000, 62, 186, 124, 0],  # 15
                    [0, 0, 62, 124, 62, 62, 2000, 62, 62, 62],  # 16
                    [0, 0, 62, 62, 0, 186, 0, 2062, 124, 0],  # 17
                    [0, 0, 0, 0, 0, 62, 62, 186, 2062, 124],  # 18
                    [0, 0, 248, 0, 62, 62, 0, 62, 62, 2000],  # 19
                    [2000, 0, 186, 0, 0, 0, 0, 62, 124, 124],  # 20
                    [0, 2000, 0, 186, 62, 0, 124, 62, 0, 62],  # 21
                    [0, 0, 2186, 0, 62, 62, 0, 62, 124, 0],  # 22
                    [0, 0, 0, 2062, 0, 62, 62, 62, 62, 186],  # 23
                ]
            elif args.num_clients == 36:
                data_distribution=[[1400,63, 7, 9,19, 2, 0, 0, 0, 0] ,
                     [ 154, 1220,85,22,15, 2, 1, 0, 1, 0] ,
                     [ 203,68, 1228, 0, 1, 0, 0, 0, 0, 0] ,
                     [ 83,30, 0, 1213, 121,19, 7,20, 2, 5] ,
                     [ 53, 229, 7, 6, 1202, 3, 0, 0, 0, 0] ,
                     [ 239,56, 1, 2, 0 ,1200, 0, 0, 1, 1] ,
                     [ 217,48,17,13, 0, 2, 1202, 0, 1, 0] ,
                     [ 176, 6,46,40,20, 0, 3,1209, 0, 0] ,
                     [ 131,41,37,79, 8, 4, 0, 0,1200, 0] ,
                     [ 127,31, 3,45,70,22, 1, 0, 0,1201] ,
                     [1300, 175, 2,16, 0, 7, 0, 0, 0, 0] ,
                     [ 186,1280,10, 6, 3,11, 1, 3, 0, 0] ,
                     [ 35, 162,1216,17,54,15, 0, 1, 0, 0] ,
                     [ 158,49,84,1205, 0, 3, 1, 0, 0, 0] ,
                     [ 237,14,19, 5,1208,10, 4, 0, 2, 1] ,
                     [ 179,58,10, 6, 1,1221,20, 5, 0, 0] ,
                     [73, 220, 7, 0, 0, 0,1200, 0, 0, 0] ,
                     [ 247,14, 0,16, 3,15, 3,1202, 0, 0] ,
                     [ 130,32,33,67,11, 5, 3,19,1200, 0] ,
                     [ 222,20, 3,55, 0, 0, 0, 0, 0,1200] ,
                     [1300, 139,46, 2, 5, 6, 2, 0, 0, 0] ,
                     [ 223,1203, 5, 0,69, 0, 0, 0, 0, 0] ,
                     [32, 170 ,1278,14, 6, 0, 0, 0, 0, 0] ,
                     [ 191,19,47 ,1226,17, 0, 0, 0, 0, 0] ,
                     [44, 118, 115,16, 1202, 4, 1, 0, 0, 0] ,
                     [ 156, 129, 7, 7, 1,1200, 0, 0, 0, 0] ,
                     [ 181,98, 8, 3,10, 0,1200, 0, 0, 0] ,
                     [ 161,55,31, 2,50, 0, 0, 1201, 0, 0] ,
                     [ 130,50,83,31, 0, 0, 4, 0, 1202, 0] ,
                     [ 262,22,15, 0, 0, 1, 0, 0, 0, 1200] ,
                     [1332,73,83,12, 0, 0, 0, 0, 0, 0] ,
                     [ 198, 1209,26,52,10, 2, 2, 0, 1, 0] ,
                     [ 288,10, 1201, 1, 0, 0, 0, 0, 0, 0] ,
                     [ 270,22, 2,1204, 2, 0, 0, 0, 0, 0] ,
                     [ 184,86,12, 0,1216, 2, 0, 0, 0, 0] ,
                     [ 124, 169, 7, 0, 0,1200, 0, 0, 0, 0] ,
                     [ 275,18, 4, 3, 0, 0,1200, 0, 0, 0] ,
                     [40,61, 155,35, 3, 1, 2,1203, 0, 0] ,
                     [ 253,45, 0, 0, 1, 0, 1, 0,1200, 0] ,
                     [ 286,12, 0, 2, 0, 0, 0, 0, 0,1200]
                ]
        elif args.data_distribution == 0.5:
            if args.num_clients == 20:
                data_distribution = [
                [1250, 0, 468, 0, 156, 312, 0, 312, 0, 0],  # 0
                [1250, 0, 312, 156, 0, 468, 156, 0, 156, 0],  # 1
                [1250, 0, 0, 156, 0, 156, 156, 156, 624, 0],  # 2
                [1250, 0, 156, 0, 0, 312, 156, 312, 156, 156],  # 3
                [1250, 0, 468, 156, 0, 156, 156, 312, 0, 0],  # 4
                [1250, 0, 0, 156, 0, 312, 156, 312, 156, 156],  # 5
                [0, 1250, 156, 0, 468, 0, 156, 468, 0, 0],  # 6f
                [0, 1250, 156, 156, 312, 156, 156, 312, 0, 0],  # 7
                [0, 1250, 156, 156, 156, 0, 468, 0, 312, 0],  # 8
                [0, 1250, 0, 312, 624, 156, 0, 0, 156, 0],  # 9
                [0, 1250, 156, 468, 0, 0, 156, 0, 0, 468],  # 10
                [0, 1250, 156, 0, 468, 156, 0, 156, 156, 156],  # 11
                [0, 0, 1562, 0, 468, 156, 156, 0, 0, 156],  # 12
                [0, 0, 156, 1874, 156, 156, 0, 0, 156, 0],  # 13
                [0, 0, 312, 156, 1250, 312, 156, 156, 0, 156],  # 14
                [0, 0, 156, 156, 312, 1250, 0, 0, 156, 468],  # 15
                [0, 0, 0, 0, 0, 468, 1562, 156, 156, 156],  # 16
                [0, 0, 156, 156, 312, 0, 312, 1250, 156, 156],  # 17
                [0, 0, 0, 312, 0, 0, 468, 156, 1406, 156],  # 18
                [0, 0, 0, 468, 0, 0, 0, 312, 468, 1250],  # 19
                [1250, 0, 156, 0, 312, 312, 0, 468, 0, 0],  # 20
                [0, 1250, 312, 156, 0, 468, 0, 156, 156, 0],  # 21
                [0, 0, 1406, 156, 156, 156, 156, 312, 156, 0],  # 22
                [0, 0, 156, 1406, 0, 312, 156, 0, 312, 156],  # 23
            ]
            elif args.num_clients == 28:
                data_distribution = [[937, 0, 0, 117, 0, 117, 234, 0, 117, 351],  # 0
                                     [937, 0, 117, 117, 117, 117, 117, 117, 0, 234],  # 1
                                     [937, 0, 117, 117, 117, 117, 234, 234, 0, 0],  # 2
                                     [937, 0, 0, 234, 117, 117, 0, 117, 234, 117],  # 3
                                     [937, 0, 117, 234, 117, 117, 117, 0, 117, 117],  # 4
                                     [937, 0, 234, 117, 0, 117, 117, 234, 117, 0],  # 5
                                     [0, 937, 234, 117, 0, 117, 0, 117, 0, 351],  # 6
                                     [0, 937, 117, 117, 0, 117, 468, 0, 117, 0],  # 7
                                     [0, 937, 0, 234, 117, 0, 117, 234, 0, 234],  # 8
                                     [0, 937, 0, 0, 234, 117, 0, 0, 468, 117],  # 9
                                     [0, 937, 234, 117, 117, 0, 117, 117, 234, 0],  # 10
                                     [0, 937, 351, 117, 0, 234, 0, 234, 0, 0],  # 11
                                     [0, 0, 1054, 117, 0, 0, 351, 117, 234, 0],  # 12
                                     [0, 0, 117, 1054, 0, 0, 351, 0, 117, 234],  # 13
                                     [0, 0, 0, 117, 1054, 117, 234, 117, 117, 117],  # 14
                                     [0, 0, 0, 585, 117, 1054, 117, 0, 0, 0],  # 15
                                     [0, 0, 0, 0, 468, 0, 1054, 117, 234, 0],  # 16
                                     [0, 0, 117, 117, 117, 0, 117, 1288, 0, 117],  # 17
                                     [0, 0, 0, 234, 0, 117, 0, 234, 1171, 117],  # 18
                                     [0, 0, 234, 0, 117, 0, 234, 0, 117, 1171],  # 19
                                     [937, 0, 234, 117, 0, 0, 0, 117, 234, 234],  # 20
                                     [0, 937, 117, 234, 0, 234, 117, 0, 117, 117],  # 21
                                     [0, 0, 1171, 0, 0, 234, 117, 0, 117, 234],  # 22
                                     [0, 0, 234, 937, 234, 117, 117, 117, 0, 117],  # 23
                                     [0, 0, 117, 117, 1054, 117, 234, 0, 117, 117],  # 24
                                     [0, 0, 117, 234, 0, 937, 351, 0, 117, 117],  # 25
                                     [0, 0, 234, 234, 0, 0, 1171, 234, 0, 0],  # 26
                                     [0, 0, 0, 117, 0, 234, 117, 1171, 0, 234],  # 27
                                     [0, 0, 234, 117, 117, 0, 234, 0, 937, 234],  # 28
                                     [0, 0, 117, 117, 117, 0, 0, 117, 468, 937],  # 29
                                     [937, 0, 0, 0, 0, 0, 0, 234, 468, 234],  # 30
                                     [0, 937, 117, 117, 117, 117, 117, 234, 117, 0]]  # 31
                # [
                # [1744,   49,   57,   17,   1,   1,   6,   0,   0,   0],
                #  [ 101,   1326,   441,   7,   0,   0,   0,   0,   0,   0],
                #  [ 382,   51,   1329,   38,   60,   12,   3,   0,   0,   0],
                #  [ 316,   600,   2,   943,   6,   3,   4,   0,   0,   1],
                #  [ 753,   182,   0,   1,   938,   0,   1,   0,   0,   0],
                #  [ 568,   117,   70,   11,   108,   970,   13,   4,   7,   7],
                #  [ 462,   157,   211,   107,   0,   0,   937,   0,   0,   1],
                #  [ 125,   420,   338,   50,   3,   0,   1,   938,   0,   0],
                #  [ 645,   81,   149,   25,   10,   5,   0,   15,   944,   1],
                #  [ 775,   16,   146,   1,   0,   0,   0,   0,   0,   937],
                #  [ 992,   664,   195,   17,   7,   0,   0,   0,   0,   0],
                #  [ 328,   1369,   16,   97,   53,   3,   3,   4,   1,   1],
                #  [ 807,   1,   1034,   10,   7,   0,   8,   0,   5,   3],
                #  [ 88,   707,   98,   955,   0,   7,   13,   0,   1,   6],
                #  [ 577,   136,   80,   21,   994,   2,   58,   3,   1,   3],
                #  [ 879,   43,   3,   13,   0,   937,   0,   0,   0,   0],
                #  [ 657,   118,   85,   8,   35,   25,   939,   4,   1,   3],
                #  [ 211,   694,   30,   1,   1,   1,   0,   937,   0,   0],
                #  [ 571,   64,   89,   182,   9,   9,   10,   0,   939,   2],
                #  [ 388,   291,   161,   96,   2,   0,   0,   0,   0,   937],
                #  [1489,   272,   98,   13,   1,   2,   0,   0,   0,   0],
                #  [ 126,   1545,   85,   119,   0,   0,   0,   0,   0,   0],
                #  [ 561,   218,   1037,   8,   44,   3,   1,   2,   0,   1],
                #  [ 130,   800,   3,   938,   4,   0,   0,   0,   0,   0],
                #  [ 379,   429,   30,   9,   961,   40,   8,   17,   0,   2],
                #  [ 839,   83,   7,   5,   1,   938,   2,   0,   0,   0],
                #  [ 925,   2,   11,   0,   0,   0,   937,   0,   0,   0],
                #  [ 636,   215,   71,   12,   1,   2,   0,   938,   0,   0],
                #  [ 781,   66,   25,   47,   8,   8,   2,   1,   937,   0],
                #  [ 641,   143,   49,   53,   28,   21,   2,   1,   0,   937],
                #  [1175,   307,   47,   177,   169,   0,   0,   0,   0,   0],
                #  [ 348,   1239,   87,   173,   23,   2,   1,   1,   0,   1]]
            elif args.num_clients+args.num_edges == 40:
                data_distribution = [[750, 0, 93, 93, 0, 279, 93, 93, 0, 93],  # 0
                                     [750, 0, 0, 186, 0, 0, 93, 93, 186, 186],  # 1
                                     [750, 0, 186, 93, 186, 93, 93, 0, 0, 93],  # 2
                                     [750, 0, 0, 0, 279, 0, 186, 0, 186, 93],  # 3
                                     [750, 0, 0, 186, 186, 186, 0, 0, 93, 93],  # 4
                                     [750, 0, 0, 93, 186, 93, 93, 0, 0, 279],  # 5
                                     [0, 750, 0, 186, 186, 0, 93, 186, 0, 93],  # 6
                                     [0, 750, 186, 0, 93, 93, 0, 93, 186, 93],  # 7
                                     [0, 750, 93, 0, 186, 0, 186, 186, 0, 93],  # 8
                                     [0, 750, 93, 93, 93, 0, 93, 93, 279, 0],  # 9
                                     [0, 750, 0, 93, 186, 93, 0, 0, 186, 186],  # 10
                                     [0, 750, 93, 186, 0, 0, 186, 0, 93, 186],  # 11
                                     [0, 0, 843, 93, 0, 186, 186, 0, 93, 93],  # 12
                                     [0, 0, 0, 843, 0, 0, 186, 93, 186, 186],  # 13
                                     [0, 0, 0, 0, 843, 186, 186, 93, 93, 93],  # 14
                                     [0, 0, 0, 93, 0, 936, 0, 93, 279, 93],  # 15
                                     [0, 0, 93, 93, 0, 279, 843, 186, 0, 0],  # 16
                                     [0, 0, 93, 93, 93, 93, 186, 843, 0, 93],  # 17
                                     [0, 0, 186, 93, 186, 0, 93, 93, 843, 0],  # 18
                                     [0, 0, 279, 93, 0, 93, 0, 93, 93, 843],  # 19
                                     [750, 0, 93, 93, 186, 0, 186, 186, 0, 0],  # 20
                                     [0, 750, 186, 93, 0, 0, 186, 0, 0, 279],  # 21
                                     [0, 0, 750, 93, 93, 93, 0, 279, 93, 93],  # 22
                                     [0, 0, 93, 843, 93, 186, 93, 93, 93, 0],  # 23
                                     [0, 0, 0, 93, 936, 93, 93, 93, 93, 93],  # 24
                                     [0, 0, 0, 0, 93, 936, 186, 93, 93, 93],  # 25
                                     [0, 0, 93, 186, 0, 0, 1029, 93, 0, 93],  # 26
                                     [0, 0, 0, 186, 93, 93, 93, 750, 186, 93],  # 27
                                     [0, 0, 93, 0, 186, 279, 0, 186, 750, 0],  # 28
                                     [0, 0, 186, 93, 93, 0, 186, 93, 0, 843],  # 29
                                     [750, 0, 93, 186, 93, 0, 186, 0, 93, 93],  # 30
                                     [0, 750, 0, 186, 279, 0, 93, 0, 0, 186],  # 31
                                     [0, 0, 843, 0, 93, 372, 0, 93, 0, 93],  # 32
                                     [0, 0, 279, 843, 93, 186, 0, 0, 0, 93],  # 33
                                     [0, 0, 0, 186, 843, 93, 93, 93, 0, 186],  # 34
                                     [0, 0, 0, 0, 93, 750, 186, 186, 93, 186],  # 35
                                     [0, 0, 93, 186, 93, 93, 750, 93, 0, 186],  # 36
                                     [0, 0, 0, 279, 93, 0, 0, 936, 93, 93],  # 37
                                     [0, 0, 0, 186, 0, 93, 186, 0, 1029, 0],  # 38
                                     [0, 0, 0, 93, 0, 186, 0, 279, 93, 843]]  # 39
        elif args.data_distribution == 0.2:
            data_distribution = [
            [500, 0, 250, 0, 0, 0, 250, 250, 750, 500],  # 0
            [500, 0, 250, 750, 0, 0, 250, 250, 500, 0],  # 1
            [500, 0, 250, 250, 250, 500, 500, 0, 0, 250],  # 2
            [500, 0, 250, 0, 750, 250, 500, 250, 0, 0],  # 3
            [500, 0, 250, 0, 250, 750, 500, 0, 0, 250],  # 4
            [500, 0, 500, 750, 250, 0, 250, 250, 0, 0],  # 5
            [0, 500, 500, 250, 0, 0, 500, 500, 0, 250],  # 6
            [0, 500, 0, 500, 500, 250, 250, 0, 500, 0],  # 7
            [0, 500, 250, 500, 250, 250, 500, 0, 0, 250],  # 8
            [0, 500, 0, 0, 0, 500, 0, 250, 750, 500],  # 9
            [0, 500, 250, 500, 500, 0, 250, 0, 500, 0],  # 10
            [0, 500, 500, 0, 500, 250, 250, 0, 0, 500],  # 11
            [0, 0, 750, 0, 0, 0, 250, 500, 1000, 0],  # 12
            [0, 0, 0, 1000, 0, 500, 0, 250, 0, 750],  # 13
            [0, 0, 0, 250, 750, 500, 0, 250, 250, 500],  # 14
            [0, 0, 250, 500, 0, 750, 250, 0, 0, 750],  # 15
            [0, 0, 500, 250, 250, 0, 500, 250, 750, 0],  # 16
            [0, 0, 250, 750, 250, 500, 0, 750, 0, 0],  # 17
            [0, 0, 0, 250, 500, 500, 500, 0, 500, 250],  # 18
            [0, 0, 750, 250, 250, 0, 250, 0, 250, 750],  # 19
            [500, 0, 500, 0, 0, 500, 250, 0, 500, 250],  # 20
            [0, 500, 250, 250, 0, 250, 0, 500, 750, 0],  # 21
            [0, 0, 750, 0, 500, 0, 250, 250, 0, 750],  # 22
            [0, 0, 250, 1000, 250, 0, 500, 500, 0, 0],  # 23
        ]

        elif args.data_distribution == 0:
            data_distribution = [
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 0
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 1
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 2
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 3
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 4
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 5
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 6
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 7
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 8
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 9
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 10
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 11
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 12
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 13
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 14
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 15
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 16
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 17
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 18
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 19
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 20
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 21
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250],  # 22
                [250, 250, 250, 250, 250, 250, 250, 250, 250, 250]  # 23
            ]
    data_distribution = np.array(data_distribution)

    data_distribution = data_distribution // 10
    data_distribution = data_distribution.tolist()
    dataloaders = get_dataloaders(args, data_distribution)

    clients_bds=data_dist.get_bandwidth_datas()

    ##########################################################################################################
    #                                         LOCAL_ML                                                       #
    ##########################################################################################################
    # all_in_one(args, data_distribution)
    ##########################################################################################################
    #                                         FL                                                             #
    ##########################################################################################################
    # compare(dataloaders, locations_list)

    ##########################################################################################################
    #                                         RL                                                             #
    ##########################################################################################################
    env = FL(args, dataloaders,clients_bds)


    agent_results = []
    plot_x = np.zeros(0)
    plot_y = np.zeros(0)
    plot_reward = np.zeros(0)
    plot_acc = np.zeros(0)
    plot_cost = np.zeros(0)
    plot_all_reward = np.zeros(0)
    plot_fake_reward = np.zeros(0)
    print("model:%s, data_dist:%.1f, target_acc:%.4f, curr_cluster:%s, num_clients:%d,device:%s"
          % (args.model, args.data_distribution, args.target_acc, args.edge_choice, args.num_clients,env.device))
    RL_path = args.RL_path
    reward_path = args.reward_path
    actor_path = args.actor_path
    LOAD = args.load
    theta=args.theta
    for run in range(RUNS):
        agent = SACAgent(env,LOAD)
        run_results = []
        for episode_number in range(EPISODES_PER_RUN):
            evaluation_episode = (episode_number % TRAINING_EVALUATION_RATIO == 0 and episode_number!=0)
            episode_reward = 0
            real_episode_reward = 0
            state = env.reset()
            done = False
            param_dic = {
                'objective_value': [],
                'variance': [],
                'accuracy': [],
                'recluster_num':0
            }
            for ep in tqdm(range(args.max_ep_step)):
                if done:
                    break
                if not LOAD:
                    action = agent.get_next_action(state, theta,evaluation_episode=evaluation_episode)
                else:
                    action = agent.get_action_deterministically(state)
                next_state, reward, real_reward, acc,param_dic, done = env.step(action,param_dic,ep)
                if ep == args.max_ep_step-1:
                    reward = reward-(args.target_acc-acc)*1000
                    print("acc: %f"%acc)
                if done:
                    print("acc: %f" % acc)
                if not evaluation_episode and not LOAD:
                    agent.train_on_transition(state, action, next_state, reward, done)
                episode_reward += reward
                real_episode_reward += real_reward
                plot_x = np.append(plot_x, args.max_ep_step * run + ep)
                plot_acc = np.append(plot_acc, acc)
                plot_reward = np.append(plot_reward, reward)
                # plot_cost = np.append(plot_cost, cost)
                np.savez(RL_path, plot_x, plot_acc, plot_reward)
                state = next_state
            if evaluation_episode:
                print(f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end=' ')
                print("episoed:%d, reward:%.5f ,real_reward:%.5f, ep:%d" %(episode_number, episode_reward, real_episode_reward,ep))
                run_results.append(episode_reward)
            if episode_number==700:
                torch.save(agent.actor_local.state_dict(), actor_path+"_ep"+str(episode_number))
            plot_y = np.append(plot_y, run)
            plot_all_reward = np.append(plot_all_reward, real_episode_reward)
            plot_fake_reward = np.append(plot_fake_reward, episode_reward)
            np.savez(reward_path, plot_y, plot_all_reward, plot_fake_reward)
            torch.save(agent.actor_local.state_dict(), actor_path)
            print("number of recluster:{}\n".format(param_dic['recluster_num']))
            with open('./result/variance_{}.json'.format(args.model), 'w') as f:
                json.dump(param_dic, f, indent=4)
        agent_results.append(run_results)
    # env.close()

    n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(len(results_mean)))
    x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]

    print("lr = 0.001")

    ax = plt.gca()
    ax.set_ylim([-30, 0])
    ax.set_ylabel('Episode Score')
    ax.set_xlabel('Training Episode')
    ax.plot(x_vals, results_mean, label='Average Result', color='blue')
    ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
    ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
    plt.legend(loc='best')
    plt.show()
