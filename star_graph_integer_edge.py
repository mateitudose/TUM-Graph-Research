import numpy as np
import matplotlib.pyplot as plt


def plotRootsOfUnity(n):
    # Compute the n-th roots of unity
    k = np.arange(n)
    roots = np.exp(2 * np.pi * 1j * k / n)

    # Create a figure with white background
    plt.figure(facecolor='white')

    # Plot the unit circle
    theta = np.linspace(0, 2 * np.pi, 1000)
    unit_circle = np.exp(1j * theta)
    plt.plot(unit_circle.real, unit_circle.imag, '--', color=[0.7, 0.7, 0.7])

    # Plot the roots of unity
    plt.scatter(roots.real, roots.imag, color='red', s=100, zorder=5)

    # Draw lines from the origin to each root
    for root in roots:
        plt.plot([0, root.real], [0, root.imag], 'r-', linewidth=1)

    # Mark the origin
    plt.scatter(0, 0, color='black', s=100, zorder=5)

    # Set plot limits and make the plot equal aspect ratio
    plt.axis('equal')
    plt.grid(True)

    # Set plot limits and labels
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'{n}-th Roots of Unity')

    # Annotate each root
    for root in roots:
        text_offset = 0.1 * root / abs(root)
        text_pos = root + text_offset
        if abs(root.imag) < 1e-10:  # Purely real
            label = f'{root.real:.2f}'
        elif abs(root.real) < 1e-10:  # Purely imaginary
            if root.imag == 1:
                label = 'i'
            elif root.imag == -1:
                label = '-i'
            else:
                label = f'{root.imag:.2f}i'
        else:  # Complex
            if root.imag > 0:
                label = f'{root.real:.2f}+{root.imag:.2f}i'
            else:
                label = f'{root.real:.2f}{root.imag:.2f}i'
        plt.text(text_pos.real, text_pos.imag, label,
                 horizontalalignment='center',
                 verticalalignment='center')

    # Show the plot
    plt.show()


# Example usage
plotRootsOfUnity(8)
