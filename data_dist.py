from FL.datasets.get_data import *
from options import args_parser

# if args.data_distribution == 0.8, 0.2 of the dataset is partitioned by labels,
# while the remaining is partitioned uniformly at random.

def suiji(x, zu):
    a = []
    a.append(random.randint(0, x))
    for i in range(1, zu - 1):
        a.append(random.randint(0, x-sum(a)))
    a.append(x - sum(a))
    return a

def data_dist():
    args = args_parser()
    dist = args.data_distribution
    num_sum = args.num_clients + args.num_edges
    # basic为每客户端数据之和
    basic = args.dataset_sample_num // num_sum

    if args.data_dist == "ICML":
        data_dist = np.full((num_sum, 10), int(basic // 10 * (1 - dist)))
        for i in range(num_sum):
            b = data_dist[i, 0:]
            s = suiji(basic * dist, 10)
            for j in range(10):
                b[j] += s[j]

    if args.data_dist == "Baochun":
        data_dist = np.full((num_sum, 10), 0)
        main_data = int(basic * dist)
        for i in range(num_sum):
            data_dist[i, i % 10] += main_data
        # print(data_dist)

        for i in range(num_sum):
            b = data_dist[i, 0:]
            s = suiji(basic - main_data, 10)
            for j in range(10):
                b[j] += s[j]

    print(data_dist)
    return data_dist
def process_log_file(filename):
    ratios = []
    with open(filename, 'r') as file:
        for line in file:
            # 按空格分割每行数据
            data = line.split()
            # 提取第 5 和第 6 列的值并做除法
            if len(data) >= 6:
                try:
                    ratio = float(data[4]) / (float(data[5])/1000)/(1024)/10
                    if ratio==0.0:
                        continue
                    ratios.append(ratio)
                except (ValueError, ZeroDivisionError):
                    pass
    return ratios
def calculate_average_ratios(ratios):
    averages = []
    for i in range(0, len(ratios), 3):
        average = np.mean(ratios[i:i+3])
        averages.append(average)
    return averages

def get_bandwidth_datas():

    clients_bds = []
    client_bd = []

    for i in range(2):
        log_file = 'data/4G/report_bicycle_000{}.log'.format(i + 1)
        ratios = process_log_file(log_file)


        # 每 3 个结果计算平均值
        for average_ratio in calculate_average_ratios(ratios):
            client_bd.append(average_ratio)
            if len(client_bd) == 50:
                clients_bds.append(client_bd)
                client_bd = []

    for i in range(2):
        path='data/4G/report_bicycle_000{}.log'.format(i+1)
        ratios = process_log_file(log_file)

        # 每 3 个结果计算平均值
        for average_ratio in calculate_average_ratios(ratios):
            client_bd.append(average_ratio)
            if len(client_bd) == 50:
                clients_bds.append(client_bd)
                client_bd = []
    for i in range(11):
        path='data/4G/report_bus_00{:02d}.log'.format(i+1)
        ratios = process_log_file(log_file)

        # 每 3 个结果计算平均值
        for average_ratio in calculate_average_ratios(ratios):
            client_bd.append(average_ratio)
            if len(client_bd) == 50:
                clients_bds.append(client_bd)
                client_bd = []

    for i in range(8):
        path='data/4G/report_car_000{}.log'.format(i+1)
        ratios = process_log_file(log_file)

        # 每 3 个结果计算平均值
        for average_ratio in calculate_average_ratios(ratios):
            client_bd.append(average_ratio)
            if len(client_bd) == 50:
                clients_bds.append(client_bd)
                client_bd = []
    for i in range(8):
        path='data/4G/report_foot_000{}.log'.format(i+1)
        ratios = process_log_file(log_file)

        # 每 3 个结果计算平均值
        for average_ratio in calculate_average_ratios(ratios):
            client_bd.append(average_ratio)
            if len(client_bd) == 50:
                clients_bds.append(client_bd)
                client_bd = []
    for i in range(3):
        path='data/4G/report_train_00{:02d}.log'.format(i+1)
        ratios = process_log_file(log_file)

        # 每 3 个结果计算平均值
        for average_ratio in calculate_average_ratios(ratios):
            client_bd.append(average_ratio)
            if len(client_bd) == 50:
                clients_bds.append(client_bd)
                client_bd = []
    for i in range(8):
        path='data/4G/report_tram_000{}.log'.format(i+1)
        ratios = process_log_file(log_file)

        # 每 3 个结果计算平均值
        for average_ratio in calculate_average_ratios(ratios):
            client_bd.append(average_ratio)
            if len(client_bd) == 50:
                clients_bds.append(client_bd)
                client_bd = []

    print(len(clients_bds))
    array_1=np.array(clients_bds)

    return clients_bds



    # print(data_distribution)
if __name__ == "__main__":
 #    data_dist = [[ 846,   601,   45,   5,   1,   0,   1,   0,   1,   0],
 # [ 591,   752,   62,   52,   32,   3,   4,   3,   1,   0],
 # [ 1,   651,   815,   3,   15,   11,   0,   1,   0,   3],
 # [ 675,   56,   18,   750,   0,   1,   0,   0,   0,   0],
 # [ 141,   268,   61,   148,   842,   31,   3,   4,   2,   0],
 # [ 569,   19,   29,   59,   11,   757,   48,   5,   0,   3],
 # [ 14,   381,   142,   203,   9,   0,   751,   0,   0,   0],
 # [ 267,   72,   361,   7,   32,   3,   1,   751,   1,   5],
 # [ 554,   22,   95,   5,   17,   3,   49,   5,   750,   0],
 # [ 171,   329,   49,   186,   14,   1,   0,   0,   0,   750],
 # [ 803,   231,   11,   60,   85,   121,   119,   59,   1,   10],
 # [ 118,   929,   359,   61,   26,   2,   5,   0,   0,   0],
 # [ 442,   215,   792,   9,   4,   36,   1,   0,   1,   0],
 # [ 521,   197,   11,   765,   5,   0,   0,   0,   0,   1],
 # [ 540,   55,   0,   131,   772,   1,   1,   0,   0,   0],
 # [ 110,   279,   46,   285,   14,   753,   2,   9,   1,   1],
 # [ 389,   54,   55,   14,   99,   57,   753,   54,   12,   13],
 # [ 445,   17,   247,   13,   22,   5,   1,   750,   0,   0],
 # [ 629,   80,   18,   17,   4,   1,   0,   1,   750,   0],
 # [ 91,   94,   257,   129,   95,   41,   11,   3,   13,   766],
 # [ 914,   318,   65,   197,   2,   0,   1,   3,   0,   0],
 # [ 223,   1106,   141,   26,   4,   0,   0,   0,   0,   0],
 # [ 579,   103,   805,   0,   1,   3,   9,   0,   0,   0],
 # [ 395,   354,   0,   751,   0,   0,   0,   0,   0,   0],
 # [ 274,   418,   36,   6,   762,   4,   0,   0,   0,   0],
 # [ 380,   190,   21,   39,   77,   778,   11,   1,   1,   2],
 # [ 93,   610,   7,   29,   6,   5,   750,   0,   0,   0],
 # [ 678,   38,   8,   13,   10,   3,   0,   750,   0,   0],
 # [ 698,   5,   25,   21,   1,   0,   0,   0,   750,   0],
 # [ 496,   81,   38,   61,   31,   9,   23,   10,   0,   751],
 # [ 788,   478,   227,   2,   2,   1,   1,   1,   0,   0],
 # [ 266,   879,   296,   31,   13,   14,   1,   0,   0,   0],
 # [ 78,   576,   810,   15,   11,   6,   4,   0,   0,   0],
 # [ 436,   193,   62,   794,   0,   5,   1,   5,   2,   2],
 # [ 29,   427,   226,   36,   768,   14,   0,   0,   0,   0],
 # [ 640,   20,   87,   3,   0,   750,   0,   0,   0,   0],
 # [ 406,   193,   48,   23,   59,   6,   759,   6,   0,   0],
 # [ 326,   230,   29,   31,   98,   4,   28,   750,   4,   0],
 # [ 209,   237,   65,   54,   35,   143,   4,   0,   751,   2],
 # [ 265,   387,   5,   43,   29,   20,   0,   1,   0,   750]]
    data_dist()


