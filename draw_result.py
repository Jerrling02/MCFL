import numpy as np
import matplotlib.pyplot as plt
import os

dir = "result"
list_path = os.listdir(dir)


def get_color_list(length):
    color_list = [
        '#19CAAD',
        '#8CC7B5',
        '#A0EEE1',
        '#BEE7E9',
        '#BEEDC7',
        '#D6D5B7',
        '#D1BA74',
        '#E6CEAC',
        '#ECAD9E',
        '#F4606C'
    ]
    start = 0
    ret_color_list = []
    for i in range(length):
        ret_color_list.append(color_list[start])
        start += 3
        start %= len(color_list)
    return ret_color_list


plot_color = get_color_list(len(list_path))




for path, color in zip(list_path, plot_color):
    if path == 'RL_reward_0.5_logistic_36client4edge.npz':
        result_file = os.path.join(dir, path)
        tmp = os.path.splitext(path)
        r = np.load(result_file)

        plot_reward = r['arr_1']
        lis = []
        for i in plot_reward:
            if i != 0:
                lis.append(i)
        print(lis)
        plot_x = range(len(lis))
        plt.title("reward")
        plt.plot(plot_x, lis, label=tmp[0], color=color)
        plt.legend()
        plt.show()

        plot_fake_reward = r['arr_2']
        lis1 = []
        for j in plot_fake_reward:
            if j != 0:
                lis1.append(j)
        print("fake_reward: ", lis1)
        plot_x1 = range(len(lis1))
        plt.title("fake reward")
        plt.plot(plot_x1, lis1, label=tmp[0], color=color)
        plt.legend()
        plt.show()



# for path, color in zip(list_path, plot_color):
#     if path == 'RL_0.5_logistic_20client4edge.npz':
#         result_file = os.path.join(dir, path)
#         tmp = os.path.splitext(path)
#         r = np.load(result_file)
#
#         max_episodes = 1
#         max_ep_step = 50
#         gap = 1
#         start = 0
#         end = -1
#         plot_x = r['arr_0']
#         plot_acc = r['arr_1']
#         plot_cost = r['arr_2']
#         plot_reward = r['arr_3']
#
#         plot_x = plot_x[start:end:gap]
#         plot_acc = plot_acc[start:end:gap]
#         plot_cost = plot_cost[start:end:gap]
#         plot_reward = plot_reward[start:end:gap]
#
#         if path == "RL.npz":
#             if plot_x.size >= max_episodes * max_ep_step:
#                 start_ = plot_x.size - plot_x.size % max_ep_step - max_episodes * max_ep_step
#                 end_ = plot_x.size - plot_x.size % max_ep_step
#                 plot_x = plot_x[0:max_episodes*max_ep_step:gap]
#                 plot_acc = plot_acc[start_:end_:gap]
#                 plot_cost = plot_cost[start_:end_:gap]
#                 plot_reward = plot_reward[start_:end_:gap]
#
#         # plot_acc = -np.log(-(plot_acc - 1))
#
#         plt.subplot(311)
#         plt.title("reward")
#         plt.plot(plot_x, plot_reward, label=tmp[0], color=color)
#         plt.legend()
#
#         plt.subplot(312)
#         plt.title("accuracy")
#         plt.plot(plot_x, plot_acc, label=tmp[0], color=color)
#         plt.legend()
#
#         plt.subplot(313)
#         plt.title("communication cost")
#         plt.plot(plot_x, plot_cost, label=tmp[0], color=color)
#         plt.legend()
#         plt.show()