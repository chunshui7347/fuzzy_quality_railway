import matplotlib.pyplot as plt


def plot_graph(type_graph, low, mid, high, labelx, labely):
    plt.plot(type_graph, low, type_graph, mid, type_graph, high)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.show()
