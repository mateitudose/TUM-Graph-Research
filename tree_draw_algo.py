from math import gcd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque

time = 0


def read_graph_console(graph, parent_list):
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


def calculate_node_levels(graph, root):
    levels = {root: 0}
    queue = deque([(root, 0)])
    while queue:
        node, level = queue.popleft()
        for neighbor in graph[node]:
            # Not already visited, so we found a node on a new level
            if neighbor not in levels:
                levels[neighbor] = level + 1
                queue.append((neighbor, level + 1))
    return levels


def get_level_order_positions(graph, root):
    # Calculate level for each node
    levels = calculate_node_levels(graph, root)
    max_level = max(levels.values())

    # Initialize level-wise counters and positions
    level_counters = defaultdict(int)
    node_positions = {}

    # Process nodes level by level
    for level in range(max_level + 1):
        # Get nodes at current level
        level_nodes = [node for node, node_level in levels.items() if node_level == level]
        # Sort nodes by their position in the tree (left to right)
        level_nodes.sort(key=lambda x: get_node_horizontal_position(graph, root, x))
        # Assign positions to nodes at this level
        for node in level_nodes:
            level_counters[level] += 1
            node_positions[node] = (level, level_counters[level] - 1)

    return node_positions


def get_node_horizontal_position(graph, root, target):
    if root == target:
        return 0

    # Just traversing the tree and passing the position of the target node which should return as a value when found
    def dfs(node, parent, pos=0):
        if node == target:
            return pos, True

        current_pos = pos
        for child in graph[node]:
            if child != parent:
                child_pos, found = dfs(child, node, current_pos)
                if found:
                    return child_pos, True
                current_pos += 1
        return pos, False

    position, _ = dfs(root, None)
    return position


def slope_translation(x_father, y_father, x_initial, y_initial):
    x_translation = x_father + x_initial
    y_translation = y_father + y_initial
    return x_translation, y_translation

def edge_intersection(existing_edges, current_edge, node_coordinates):
    # function returns false if there are no intersections and true otherwise
    current_parent, current_child = current_edge[:2]
    x1, y1 = node_coordinates[current_parent][:2]
    x2, y2 = node_coordinates[current_child][:2]
    #if there are no edges in the graph there will be no intersections with the current edge
    if not existing_edges:
        return False
    #check if the edges assigned intersect with the current_edge
    for existing_edge in existing_edges:
        existing_parent, existing_child = existing_edge[:2]
        x3, y3 = node_coordinates[existing_parent][:2]
        x4, y4 = node_coordinates[existing_child][:2]

        # Skip if edges share a vertex - they can't intersect
        if (current_parent == existing_parent or
                current_parent == existing_child or
                current_child == existing_parent or
                current_child == existing_child):
            continue

        # Calculate the denominator for intersection check
        denominator = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        # If lines are parallel
        if denominator == 0:
            continue

        # Calculate intersection parameters
        ua = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

        # Check if intersection occurs within both line segments
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return True
    return False


def no_intersections(existing_edges, new_edge, node_coordinates, used_slopes_on_level, level, slope_index):
    # if there are no intersections
    if not edge_intersection(existing_edges, new_edge, node_coordinates):
        existing_edges.append(new_edge)  # add the new edge to the graph
        used_slopes_on_level[level].append(slope_index)  # mark the slope as used on this level
        return True
    return False

def calculate_nodes_coords(graph, root, current_node, node_coordinates, parent_list, triplets, discovery_time,
                           visited, node_positions, existing_edges, used_slopes_on_level):
    global time
    if current_node != root:
        visited.add(current_node)
        discovery_time[current_node] = time
        # Get slope index based on node position or discovery time
        level, position = node_positions[current_node]
        if level == 0:  # Using discovery time for non-reuse mode
            slope_index = time
        else:  # Using position for reuse mode
            slope_index = position

        slope_x, slope_y = triplets[slope_index][:2]  # initial choice for slope
        time += 1
        parent = parent_list[current_node]
        parent_x, parent_y = node_coordinates[parent][:2]
        node_coordinates[current_node] = slope_translation(parent_x, parent_y, slope_x, slope_y)  # initial translation
        if level != 0:
            new_edge = (parent, current_node)
            # if there are intersections
            if not no_intersections(existing_edges, new_edge, node_coordinates, used_slopes_on_level, level,
                                    slope_index):
                # find a new slope
                for index, slope in enumerate(triplets):
                    slope_x, slope_y = triplets[index][:2]
                    # if this slope was not used on the level
                    if index not in used_slopes_on_level[level]:
                        node_coordinates[current_node] = slope_translation(parent_x, parent_y, slope_x, slope_y)
                        #if a suitable slope is found we escape the for
                        if no_intersections(existing_edges, new_edge, node_coordinates, used_slopes_on_level, level,
                                            index):
                            break
    for neighbour in graph[current_node]:
        if neighbour not in visited:
            calculate_nodes_coords(graph, root, neighbour, node_coordinates, parent_list, triplets,
                                   discovery_time, visited, node_positions, existing_edges, used_slopes_on_level)

def draw_tree():
    parent_list = {1: 1}
    visited = set()
    discovery_time = {}
    node_coordinates = {1: (0, 0)}
    subtree_sizes = {}
    root = 1
    existing_edges = []
    visited.add(root)
    input_graph = nx.Graph()
    final_graph = nx.Graph()
    traversal_algorithm, should_reuse_slopes = read_graph_console(input_graph, parent_list)

    if traversal_algorithm == "WDFS":
        calculate_subtree_sizes(input_graph, root, None, subtree_sizes)
        print(subtree_sizes)
        for node in input_graph.nodes:
            sorted_neighbors = sorted(input_graph[node], key=lambda x: subtree_sizes[x], reverse=True)
            final_graph.add_edges_from((node, neighbor) for neighbor in sorted_neighbors)
    else:
        final_graph = input_graph

    # Calculate node positions based on level-order traversal if reusing slopes
    node_positions = get_level_order_positions(final_graph, root) if should_reuse_slopes else {
        node: (0, time) for time, node in enumerate(range(1, len(final_graph.nodes) + 1))
    }

    # Generate Pythagorean triplets
    if should_reuse_slopes:
        # For slope reuse: generate based on max nodes per level
        max_nodes_per_level = max(sum(1 for x in node_positions.values() if x[0] == level)
                                  for level in range(max(x[0] for x in node_positions.values()) + 1))
        #generate more slopes in case we have some intersections
        triplets = generate_pythagorean_triplets(max_nodes_per_level * 2)
    else:
        # For unique slopes: generate based on total nodes minus root
        triplets = generate_pythagorean_triplets(len(final_graph.nodes) - 1)
    if triplets:
        triplets.sort(key=lambda x: x[1] / x[0])
    print(len(triplets))
    max_level = max(node_positions[node][0] for node in node_positions)
    used_slopes_on_level = [[] for _ in range(max_level + 1)]

    # Calculate coordinates with reused slopes
    calculate_nodes_coords(final_graph, root, root, node_coordinates, parent_list, triplets,
                           discovery_time, visited, node_positions, existing_edges, used_slopes_on_level)

    # Draw the tree
    fig, ax = plt.subplots()
    fig.set_dpi(200)
    ax.set_aspect('equal')
    ax.set_axis_on()

    for node in node_coordinates:
        current_coords = node_coordinates[node]
        if parent_list[node] == root:
            ax.plot([0, current_coords[0]], [0, current_coords[1]], marker='o', markersize=5, color='black')
        else:
            parent_coords = node_coordinates[parent_list[node]]
            ax.plot([parent_coords[0], current_coords[0]], [parent_coords[1], current_coords[1]],
                    marker='o', markersize=5, color='black')

    for node in node_coordinates:
        current_coords = node_coordinates[node]
        ax.text(current_coords[0], current_coords[1], str(node), fontsize=12, ha='right', va='bottom')

    plt.show()
    # Print the node coordinates and positions just for debugging purposes
    print("Node coordinates:", node_coordinates)
    print("Node positions by level:", node_positions)

if __name__ == "__main__":
    draw_tree()
