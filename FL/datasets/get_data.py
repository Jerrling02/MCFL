import torch

"""
download the required dataset, split the data among the clients, and generate DataLoader for training
"""
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
cudnn.banchmark = True

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import random

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target

def split_data(dataset, args, kwargs, data_distribution, is_shuffle = True):
    data_loaders = [0] * (args.num_clients + args.num_edges)
    dict_users = {i: np.array([]) for i in range(args.num_clients + args.num_edges)}
    idxs = np.arange(len(dataset))
    # is_shuffle is used to differentiate between train and test
    labels = dataset.targets
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    # sort the data according to their label
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    

    # for i in range(2):
    #     alloc_list = tmp[i]
    #     for digit, num_of_digit in enumerate(alloc_list):
    #         tmp1 = np.argwhere(idxs_labels[1, :] == digit)
    #         tmp1 = tmp1.ravel()
    #         tmp2 = np.random.choice(idxs_labels[0, tmp1[0:1]], num_of_digit, replace = True)
    #         dict_users[i] = np.concatenate((dict_users[i], tmp2), axis=0)
    #         dict_users[i] = dict_users[i].astype(int)
    #     data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
    #                                 batch_size = args.batch_size,
    #                                 shuffle = is_shuffle, **kwargs)
    # tmp = np.array(tmp)
    # tmp *= 40
    # tmp = tmp.tolist()
    for i in range(args.num_clients + args.num_edges):
        alloc_list = data_distribution[i]
        for digit, num_of_digit in enumerate(alloc_list):
            tmp1 = np.argwhere(idxs_labels[1, :] == digit)
            tmp1 = tmp1.ravel()
            tmp2 = np.random.choice(idxs_labels[0, tmp1], num_of_digit, replace=True)
            dict_users[i] = np.concatenate((dict_users[i], tmp2), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size=args.batch_size,
                                    shuffle=True, **kwargs)
    return data_loaders

def get_mnist(data_distribution, dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    # num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
    # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=True,
                            download=True, transform=transform)
    test = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=False,
                            download=True, transform=transform)
    # note: is_shuffle here also is a flag for differentiating train and test
    train_loaders = split_data(train, args, kwargs, data_distribution, is_shuffle=True)
    test_loaders = split_data(test, args, kwargs, data_distribution, is_shuffle=False)
    # the actual batch_size may need to change.... Depend on the actual gradient...
    # originally written to get the gradient of the whole dataset
    # but now it seems to be able to improve speed of getting accuracy of virtual sequence
    v_train_loader = DataLoader(train, batch_size = args.batch_size * (args.num_clients + args.num_edges),
                                shuffle = True, **kwargs)
    v_test_loader = DataLoader(test, batch_size = args.batch_size * (args.num_clients + args.num_edges),
                                shuffle = False, **kwargs)
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_dataloaders(args, data_distribution):
    """
    :param args:
    :return: A list of trainloaders, a list of testloaders, a concatenated trainloader and a concatenated testloader
    """

    train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(data_distribution=data_distribution, dataset_root='data', args=args)
    # 为所有参与训练的edge和client分训练数据集和测试数据集
    # print("loading dataset for all client")
    train_loaders_ = [[] for i in range(args.num_clients + args.num_edges)]
    test_loaders_ = [[] for i in range(args.num_clients + args.num_edges)]
    v_test_loader_ = []  
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    for i in range(args.num_clients + args.num_edges):
        if i < args.num_clients:
            print("loading dataset for client", i)
        else:
            print("loading dataset for edge", i - args.num_clients)
        for data in train_loaders[i]:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            data = inputs, labels
            train_loaders_[i].append(data)
        for data in test_loaders[i]:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            data = inputs, labels
            test_loaders_[i].append(data)

    for data in v_test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        data = inputs, labels
        v_test_loader_.append(data)
    return train_loaders_, test_loaders_, v_train_loader, v_test_loader_, data_distribution
    # return train_loaders, test_loaders, v_train_loader, v_test_loader,data_distribution