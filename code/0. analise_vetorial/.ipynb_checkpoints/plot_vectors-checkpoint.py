import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_vector_sum(a, b):
    # Compute vector sum
    s = a + b

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original vectors
    ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', label='a')
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color='g', label='b')

    # Plot the sum vector
    ax.quiver(0, 0, 0, s[0], s[1], s[2], color='b', label='a + b')

    # (Optional) Show parallelogram by shifting b
    ax.quiver(a[0], a[1], a[2], b[0], b[1], b[2],
              color='gray', linestyle='dashed', alpha=0.5)
    ax.quiver(b[0], b[1], b[2], a[0], a[1], a[2],
              color='gray', linestyle='dashed', alpha=0.5)

    # Labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    max_range = np.max(np.abs([a, b, s]))
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])

    ax.legend()
    plt.show()

def plot_cross_product(a, b):
    # Compute cross product
    cross = np.cross(a, b)

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot vectors
    ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', label='a')
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color='g', label='b')
    ax.quiver(0, 0, 0, cross[0], cross[1], cross[2], color='b', label='a × b')

    # Set axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set axis limits automatically
    max_range = np.max(np.abs([a, b, cross]))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.legend()
    plt.show()

def plot_dot_product(a, b):
    # Dot product
    dot = np.dot(a, b)

    # Projection of a onto b
    b_norm = b / np.linalg.norm(b)
    proj_a_on_b = np.dot(a, b_norm) * b_norm

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot vectors
    ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', label='a')
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color='g', label='b')
    ax.quiver(0, 0, 0, proj_a_on_b[0], proj_a_on_b[1], proj_a_on_b[2],
              color='b', linestyle='dashed', label='Projection of a on b')

    # Connect projection foot to vector a
    ax.plot([a[0], proj_a_on_b[0]],
            [a[1], proj_a_on_b[1]],
            [a[2], proj_a_on_b[2]], 'k--', alpha=0.6)

    # Labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    max_range = np.max(np.abs([a, b]))
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])

    ax.legend()
    plt.title(f"Dot product: a · b = {dot:.2f}")
    plt.show()