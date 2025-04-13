import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D

# Define B(1,2) and B(1,3)
sqrt2_inv = 1 / np.sqrt(2)
B12 = sqrt2_inv * np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 0]
])
B13 = sqrt2_inv * np.array([
    [0, 0, 1],
    [0, 0, 0],
    [-1, 0, 0]
])
A = B12 + B13

# Reference vectors
north_pole = np.array([1.0, 0.0, 0.0])
SEA = np.array([0.0, 1.0, 0.0])
africa = np.array([0.0, 0.0, 1.0])

# Compute trajectory
timesteps = np.linspace(0, 2 * np.pi, 200)
trajectory = np.array([expm(t * A) @ north_pole for t in timesteps])

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot a transparent unit sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(x, y, z, color='lightgray', alpha=0.1, edgecolor='none')

# Plot the trajectory
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color='crimson', linewidth=2, label='Trajectory of North Pole')

# Plot reference points
ax.scatter(*north_pole, color='black', s=60, label='North Pole')
ax.scatter(*SEA, color='blue', s=60, label='SEA')
ax.scatter(*africa, color='green', s=60, label='Africa')

# Add text labels
ax.text(*north_pole, '  North Pole', color='black')
ax.text(*SEA, '  SEA', color='blue')
ax.text(*africa, '  Africa', color='green')

# Set axis limits and view
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=25, azim=45)
ax.set_title("3D Trajectory on the Unit Sphere under $\\exp(t(B_{1,2} + B_{1,3}))$")
ax.legend()
plt.tight_layout()
plt.show()
