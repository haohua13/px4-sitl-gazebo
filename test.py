import numpy as np
import math
import matplotlib.pyplot as plt
image_width = 720
image_height = 480
mid_x = image_width // 2
mid_y = image_height // 2
a = 200
b = 400
N = 200
# Generate random x and y coordinates between a and b
# Generate random integers between a and b (inclusive)
points_x = np.random.randint(a, b + 1, size=N)  # x-coordinates between a and b
points_y = np.random.randint(a, b + 1, size=N)  # y-coordinates between a and b
# Stack the points to form a 2xN matrix
points = np.vstack((points_x, points_y))

# Stack the points to form a 2xN matrix and then transpose it
points = np.vstack((points_x, points_y)).T

# Convert the points to the desired format
p0 = points.tolist()

print("Points in the desired format:")
print(p0)

N = 200
radius = 200
center_x = image_width // 2
center_y = image_height // 2

rho = np.sqrt(np.random.uniform(0, radius, N))
phi = np.random.uniform(0, 2*np.pi, N)

x = rho * np.cos(phi) + center_x
y = rho * np.sin(phi) + center_y
points = np.vstack((x, y)).T
print(points)


def generate_inside_points(N, phi_0, phi_1, theta_0, theta_1):
    # Generate random points inside the defined area of integration
    phi_points = np.random.uniform(phi_0, phi_1, N)
    theta_points = np.random.uniform(theta_0, theta_1, N)
    x = ((np.sin(theta_points) * np.cos(phi_points) + 1) * (image_width - 1) / 2)
    y = ((np.sin(theta_points) * np.sin(phi_points) + 1 )* (image_height - 1) / 2)
    points = np.unique(np.round(np.vstack((x, y)).T), axis = 0)
    return points

# Example usage
N = 100  # Number of points to generate
phi_0 = 0  # Minimum phi value
phi_1 = 2*np.pi  # Maximum phi value
theta_0 = 0.7*np.pi/12  # Minimum theta value
theta_1 = 2*np.pi/12  # Maximum theta value

# Plot the points as a scatter plot
plt.scatter(points[:, 0], points[:, 1], marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of points')
plt.show()

# Parameters
image_width = 720
image_height = 480

h_fov = 1.5708
f = (image_width)/(2*math.tan(h_fov/2))
ox = image_width/2
oy = image_height/2

# Camera intrinsic matrix K
K = np.array([
    [-f , 0, ox],
    [0, -f, oy],
    [0, 0, 1]
])
print(K)
print(np.linalg.inv(K))

for k in range(10):
    for i in range(20):
        for j in range(50):
            if j == 4:
                break
        print(j)
        print(i)
        print(k)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a unit sphere mesh
phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)
phi, theta = np.meshgrid(phi, theta)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Plot the unit sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.5)
u = 1.0  # Replace with your actual values
v = 2.0
vector_magnitude = np.sqrt(u**2 + v**2)
normalized_vector = np.array([u / vector_magnitude, v / vector_magnitude, 0])
# Plot the normalized vector
ax.quiver(0, 0, 1, normalized_vector[0], normalized_vector[1], normalized_vector[2], color='r', label='Vector Projection')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.legend()

plt.show()