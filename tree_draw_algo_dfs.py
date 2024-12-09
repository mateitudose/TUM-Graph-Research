import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import gcd

time = 0

def read_graph_console(graph, parent_list):
    vertices = int(input("Enter the number of vertices:\n"))
    print("Enter the edges:\n")
    for i in range(vertices - 1):
        u, v = map(int, input().split())
        graph.add_edge(u, v)
        parent_list[v] = u


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


# def calculate_subtree_sizes(graph, root, parent, subtree_sizes):
#     size = 1
#     for child in graph[root]:
#         if child != parent:
#             size += calculate_subtree_sizes(graph, child, root, subtree_sizes)
#     subtree_sizes[root] = size
#     return size


def slope_translation(x_father, y_father, x_initial, y_initial):
    x_translation = x_father + x_initial
    y_translation = y_father + y_initial
    return x_translation, y_translation


def calculate_nodes_coords(graph, root, current_node, node_coordinates, parent_list, triplets, discovery_time,
                           visited):
    global time
    if current_node != root:
        visited.add(current_node)
        discovery_time[current_node] = time
        slope_x, slope_y = triplets[time][:2]
        time += 1
        if parent_list[current_node] == root:
            node_coordinates[current_node] = (slope_x, slope_y)
        else:
            parent = parent_list[current_node]
            parent_x, parent_y = node_coordinates[parent][:2]
            node_coordinates[current_node] = slope_translation(parent_x, parent_y, slope_x, slope_y)
    for neighbour in graph[current_node]:
        if neighbour not in visited:
            calculate_nodes_coords(graph, root, neighbour, node_coordinates, parent_list, triplets,
                                   discovery_time, visited)


def draw_tree():
    parent_list = {1: 1}
    visited = set()
    discovery_time = {}
    node_coordinates = {1: (0, 0)}
    subtree_sizes = {}
    root = 1
    graph = nx.Graph()
    read_graph_console(graph, parent_list)
    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos, with_labels=True)
    # plt.show()

    # Calculate the subtree sizes of the tree
    # calculate_subtree_sizes(graph, root, None, subtree_sizes)
    # print(subtree_sizes)

    # Generate the Pythagorean triplets
    triplets = generate_pythagorean_triplets(len(graph.nodes) - 1)
    triplets.sort(key=lambda x: x[0] / x[1])
    print(triplets)

    # Calculate the coordinates of the nodes
    calculate_nodes_coords(graph, root, root, node_coordinates, parent_list, triplets, discovery_time, visited)

    # Draw the tree
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_axis_off()
    for node in node_coordinates:
        current_coords = node_coordinates[node]
        if parent_list[node] == root:
            # Draw a line from (0, 0) to the node's coordinates
            ax.plot([0, current_coords[0]], [0, current_coords[1]], marker='o', markersize=5, color='black')
        else:
            parent_coords = node_coordinates[parent_list[node]]
            # Draw a line from parent's coordinates to the current node's coordinates
            ax.plot([parent_coords[0], current_coords[0]], [parent_coords[1], current_coords[1]], marker='o',
                    markersize=5, color='black')

    for node in node_coordinates:
        current_coords = node_coordinates[node]
        ax.text(current_coords[0], current_coords[1], str(node), fontsize=12, ha='right', va='bottom')

    plt.show()
    print(node_coordinates)


if __name__ == "__main__":
    draw_tree()
