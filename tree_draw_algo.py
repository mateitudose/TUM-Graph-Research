import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import gcd

def read_graph_console(graph):
    vertices = int(input("Enter the number of vertices:\n"))
    print("Enter the edges:\n")
    for i in range(vertices - 1):
        u, v = map(int, input().split())
        graph.add_edge(u, v)


def generate_pythagorean_triplets(first_n):
    triplets = []
    m = 2

    while len(triplets) < first_n:
        for n in range(1, m):
            if (m - n) % 2 == 1 and gcd(m, n) == 1:
                a = m ** 2 - n ** 2
                b = 2 * m * n
                c = m ** 2 + n ** 2
                triplets.append((a, b, c))
                if len(triplets) == first_n:
                    return triplets
        m += 1

def calculate_subtree_sizes(graph, root, parent, subtree_sizes):
    size = 1
    for child in graph[root]:
        if child != parent:
            size += calculate_subtree_sizes(graph, child, root, subtree_sizes)
    subtree_sizes[root] = size
    return size

def draw_tree():
    graph = nx.Graph()
    read_graph_console(graph)
    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos, with_labels=True)
    # plt.show()

    # Calculate the subtree sizes of the tree
    subtree_sizes= {}
    root = 1
    calculate_subtree_sizes(graph, root, None, subtree_sizes)
    print(subtree_sizes)

    # Generate the Pythagorean triplets
    triplets = generate_pythagorean_triplets(subtree_sizes[root] - 1)



if __name__ == "__main__":
    draw_tree()
