import networkx as nx
import pylab as plt
from math import atan, gcd


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


def dfs(G: nx.Graph, root):
    stack = [root]
    visited = {root}

    for index in range(G.number_of_nodes()):
        u = stack.pop()
        G.nodes[u]['order'] = index

        neighbors = [v for v in G.neighbors(u) if v not in visited]
        for v in neighbors:
            visited.add(v)
            stack.append(v)
            G.nodes[v]['parent'] = u


def weighted_dfs(G: nx.Graph, root):
    pass


def assign_pos(G: nx.Graph, root):
    n = G.number_of_nodes()

    triplets = generate_pythagorean_triplets(n - 1)
    triplets.sort(key=lambda x: atan(x[1] / x[0]), reverse=True)

    process_order = sorted((v for v in G.nodes()), key=lambda v: G.nodes[v]['order'])

    pos = dict()
    pos[root] = {'x': 0, 'y': 0}

    for (a, b, c), v in zip(triplets, process_order[1:]):
        parent = G.nodes[v]['parent']
        pos[v] = {
            'x': pos[parent]['x'] + a,
            'y': pos[parent]['y'] + b
        }

    return pos


def tree_draw(G: nx.Graph, root, output=None):
    dfs(G, root)
    pos = assign_pos(G, root)

    fig, ax = plt.subplots()

    for u, v in G.edges():
        ax.plot(
            [pos[u]['x'], pos[v]['x']],
            [pos[u]['y'], pos[v]['y']],
            color='black'
        )

    if output is None:
        plt.show()
    else:
        fig.set_dpi(100)
        fig.savefig(output)

    plt.close(fig)


if __name__ == "__main__":
    G = nx.random_tree(15)
    root = 1

    tree_draw(G, root)
