import matplotlib.pyplot as plt

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def pythagoreanTriplets(limits):
    triplets = []
    c, m = 0, 2

    while c < limits:
        for n in range(1, m):
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n

            if c > limits:
                break

            if gcd(a, b) == 1:
                triplets.append((a, b, c))

        m = m + 1

    return triplets

def plotTriplets(triplets, n):
    plt.figure(facecolor='white')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    points_plotted = 0
    for triplet in triplets:
        if points_plotted >= n:
            break
        a, b, _ = triplet
        points = [(a, b), (-a, b), (a, -b), (-a, -b), (b, a), (-b, a), (b, -a), (-b, -a)]
        for x, y in points:
            if points_plotted >= n:
                break
            plt.plot([0, x], [0, y], marker='o')
            plt.text(x, y, f'({x}, {y})', fontsize=9, ha='right')
            points_plotted += 1

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'{n}-star graph with integer edges and coordinates')
    plt.show()

triplets = pythagoreanTriplets(105)
plotTriplets(triplets, n=20)
