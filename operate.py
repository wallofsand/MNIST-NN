import numpy as np
from matplotlib import pyplot as plt
def operate(fname='output.csv'):
    # open the test results from csv format
    with open(fname) as f:
        # data order is loc, scale, E1, E2, E3, E4, test
        data = np.genfromtxt(f, delimiter=',', skip_header=1)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    cidx = 0
    for row in data:
        x = []
        y = []
        labels = []
        color_list = []
        labels.append('alpha: {:0.3f}'.format(row[2]))
        for idx in range(1, 6):
            x.append(idx)
            y.append(row[idx + 2])
            color_list.append(colors[cidx])
        cidx = (cidx + 1) % len(colors)
        plt.scatter(x, y, c=color_list, label=labels)
    plt.title("hyperparameter test results")
    plt.xlabel("Epoch 1/4 - 2/4 - 3/4 - 4/4 - test")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

operate()
