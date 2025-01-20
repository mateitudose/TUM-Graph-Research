from math import gcd
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque, defaultdict

time = 0

def find_farthest_node(graph, start):
    """Find the node farthest from the start node using BFS."""
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start] = 0
    queue = deque([(start, 0)])

    while queue:
        node, dist = queue.popleft()
        for neighbor in graph[node]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))

    max_dist_node = max(distances.items(), key=lambda x: x[1])[0]
    return max_dist_node, distances[max_dist_node]

def find_diameter_and_center(graph):
    """Find the diameter of the graph and its center."""
    # Start from any node to find one end of diameter
    end1, _ = find_farthest_node(graph, list(graph.nodes())[0])

    # Find the other end of diameter

    end2, diameter = find_farthest_node(graph, end1)

    # Find the path of the diameter
    path = nx.shortest_path(graph, end1, end2)

    # The center is the middle node(s) of the diameter path
    center = path[len(path)//2]

    return center

def read_graph_console(graph, parent_list, has_child):
    strategy = str(input("Enter the graph traversal algorithm choice (DFS or WDFS):\n"))
    if (strategy != "DFS") and (strategy != "WDFS"):
        print("Invalid graph traversal algorithm choice!")
        exit(1)

    reuse_slopes = input("Would you like to reuse slopes for nodes at same positions? (y/n):\n").lower()
    if reuse_slopes not in ['y', 'n']:
        print("Invalid choice for slope reuse!")
        exit(1)

    vertices = int(input("Enter the number of vertices:\n"))
    print("Enter the edges:\n")
    for i in range(vertices - 1):
        u, v = map(int, input().split())
        graph.add_edge(u, v)
        parent_list[v] = u
        has_child[u] = True
    return strategy, (reuse_slopes == 'y')


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


def calculate_number_levels_subtree(graph, current_root, parent_list, number_levels_subtree, visited=None):
    # Initialize visited set on first call
    if visited is None:
        visited = set()
    # If we've already visited this node
    if current_root in visited:
        return 0
    # Mark current node as visited
    visited.add(current_root)
    # If there are no children we consider the root on level 1
    if not graph[current_root]:
        number_levels_subtree[current_root] = 1
        return 1
    number_levels = 0
    for child in graph[current_root]:
        # Skip the parent to avoid going back up the tree
        if child in parent_list and parent_list[child] == current_root:
            continue
        # Calculate max depth among all children
        child_levels = calculate_number_levels_subtree(graph, child, parent_list, number_levels_subtree, visited)
        number_levels = max(number_levels, child_levels)
    # Store and return result
    number_levels_subtree[current_root] = number_levels + 1
    return number_levels + 1

def calculate_slopes_number(graph, has_child):
    initial_number_slopes = len(graph.nodes)
    for node in graph.nodes:
        # If the node has children
        if has_child[node]:
            print(graph[node])
            # We extract 1 from the number of slopes, because one child would reuse the same slope as the parent
            initial_number_slopes -= 1
    print(f"has slopes: {initial_number_slopes}")
    return initial_number_slopes

def slope_translation(x_father, y_father, x_initial, y_initial):
    x_translation = x_father + x_initial
    y_translation = y_father + y_initial
    return x_translation, y_translation


def calculate_nodes_coords(graph, root, current_node, node_coordinates, parent_list, triplets, discovery_time,
                           visited, should_reuse_slopes, slope_assigned, greatest_index=-1):
    if not should_reuse_slopes:
        global time
        if current_node != root:
            visited.add(current_node)
            discovery_time[current_node] = time
            slope_x, slope_y = triplets[time][:2]
            time += 1
            parent = parent_list[current_node]
            parent_x, parent_y = node_coordinates[parent][:2]
            node_coordinates[current_node] = slope_translation(parent_x, parent_y, slope_x, slope_y)
        for neighbour in graph[current_node]:
            if neighbour not in visited:
                parent_list[neighbour] = current_node
                calculate_nodes_coords(graph, root, neighbour, node_coordinates, parent_list, triplets, discovery_time,
                                       visited, should_reuse_slopes, slope_assigned, greatest_index)
    else:
        if current_node == root:
            slope_assigned[current_node] = 0
        if current_node != root:
            visited.add(current_node)
            parent = parent_list[current_node]
            parent_x, parent_y = node_coordinates[parent][:2]
            # If it is the first child of its parent
            #print(f"leftmost child of 2 is {list(graph[2])[0]}")
            # The leftmost child would be on position 1, because on position 0 we have its father who will have more levels
            if list(graph[parent])[1] == current_node and parent in slope_assigned:
                slope_assigned[current_node] = slope_assigned[parent]
                slope_x, slope_y = triplets[slope_assigned[current_node]][:2]
                node_coordinates[current_node] = slope_translation(parent_x, parent_y, slope_x, slope_y)
            else:
                for node in graph:
                    if node in slope_assigned and slope_assigned[node] > greatest_index:
                        greatest_index = slope_assigned[node]
                slope_assigned[current_node] = greatest_index + 1
                slope_x, slope_y = triplets[slope_assigned[current_node]][:2]
                node_coordinates[current_node] = slope_translation(parent_x, parent_y, slope_x, slope_y)
        for neighbour in graph[current_node]:
            if neighbour not in visited:
                parent_list[neighbour] = current_node
                calculate_nodes_coords(graph, root, neighbour, node_coordinates, parent_list, triplets,
                                       discovery_time,
                                       visited, should_reuse_slopes, slope_assigned, greatest_index)

def draw_tree():
    input_graph = nx.Graph()
    has_child = defaultdict(bool)
    traversal_algorithm, should_reuse_slopes = read_graph_console(input_graph, {}, has_child)

    # Find the center of the graph to use as root
    root = find_diameter_and_center(input_graph)

    # Initialize data structures
    parent_list = {root: root}
    visited = set()
    discovery_time = {}
    node_coordinates = {root: (0, 0)}
    subtree_sizes = {}
    # Don't forget to add the root to the visited set!
    visited.add(root)
    final_graph = nx.Graph()

    if traversal_algorithm == "WDFS" and not should_reuse_slopes:
        # Calculate the subtree sizes of the tree
        calculate_subtree_sizes(input_graph, root, None, subtree_sizes)
        print(subtree_sizes)
        # Sort each node's neighbors by their subtree sizes in descending order
        # Basically, we want to visit the nodes with the largest subtrees first
        for node in input_graph.nodes:
            sorted_neighbors = sorted(input_graph[node], key=lambda x: subtree_sizes.get(x, 0), reverse=True)
            final_graph.add_edges_from((node, neighbor) for neighbor in sorted_neighbors)

    # We keep the original graph if the strategy is DFS
    elif traversal_algorithm == "DFS" and not should_reuse_slopes:
        final_graph = input_graph
    #Consider each subtree at first with 0 levels
    number_levels_subtree = {}
    slope_assigned = {}
    if should_reuse_slopes:
        calculate_number_levels_subtree(input_graph, root, parent_list, number_levels_subtree)
        for node in input_graph.nodes:
            # Sort each node's children by the number of levels of their subtree in descending order
            sorted_children = sorted(input_graph[node],key=lambda x: number_levels_subtree[x], reverse=True)
            final_graph.add_edges_from((node, child) for child in sorted_children)
    # Generate the Pythagorean triplets
    if not should_reuse_slopes:
        triplets = generate_pythagorean_triplets(len(final_graph.nodes) - 1)
    # If I want to reuse slopes
    else:
        number_slopes = calculate_slopes_number(final_graph, has_child)
        triplets = generate_pythagorean_triplets(number_slopes)
    # IMPORTANT! Sort the slopes by their increasing angle size with the x-axis
    triplets.sort(key=lambda x: x[1] / x[0])
    # print(triplets)
    # Calculate the coordinates of the nodes
    calculate_nodes_coords(final_graph, root, root, node_coordinates, parent_list, triplets, discovery_time, visited, should_reuse_slopes, slope_assigned, 0)

    # Draw the tree
    fig, ax = plt.subplots()
    fig.set_dpi(200)
    ax.set_aspect('equal')
    # ax.set_axis_off()

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
    print(f"Root node (center of diameter): {root}")
    print(node_coordinates)


if __name__ == "__main__":
    draw_tree()
